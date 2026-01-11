'use client'

import { useState, useCallback, useEffect } from 'react'
import ImageUploader from '@/components/ImageUploader'
import MaskEditor from '@/components/MaskEditor'
import ResultDisplay from '@/components/ResultDisplay'
import Gallery from '@/components/Gallery'
import Settings from '@/components/Settings'
import {
  checkV2Available,
  getSystemStatus,
  getProfiles,
  generateAdvancedTryon,
  extractFeatures,
  type ModelProfile,
  type FeatureExtractionResult,
} from '@/lib/api'

type View = 'tryon' | 'gallery' | 'catalog' | 'settings'
type Step = 'upload' | 'mask' | 'generate' | 'result'

interface GenerationSettings {
  prompt: string
  negativePrompt: string
  steps: number
  cfgScale: number
  sampler: string
  denoise: number
  model: string
  // V2 Advanced settings
  preserveFace: boolean
  preserveSkinTone: boolean
  upscale: boolean
  faceRestore: boolean
  selectedProfileId: string
}

const DEFAULT_SETTINGS: GenerationSettings = {
  prompt: 'person wearing elegant garment, professional fashion photography, high quality, detailed fabric texture',
  negativePrompt: 'blurry, distorted, low quality, deformed, bad anatomy, ugly, disfigured',
  steps: 15,
  cfgScale: 7.0,
  sampler: 'euler_ancestral',
  denoise: 0.85,
  model: 'realisticVision',
  // V2 defaults
  preserveFace: true,
  preserveSkinTone: true,
  upscale: false,
  faceRestore: true,
  selectedProfileId: '',
}

// Preset prompts for different garment categories
const PROMPT_PRESETS = {
  everyday: 'person wearing casual everyday clothing, natural pose, professional photo',
  formal: 'person wearing elegant formal attire, sophisticated, professional fashion photography',
  swimwear: 'person wearing stylish swimwear, beach setting, high quality fashion photo',
  lingerie: 'person wearing delicate lingerie, elegant boudoir photography, soft lighting',
  sheer: 'person wearing sheer fabric garment, artistic fashion photography, detailed texture',
  bridal: 'person wearing beautiful bridal wear, romantic, professional wedding photography',
  activewear: 'person wearing athletic activewear, dynamic pose, sports photography',
}

export default function Home() {
  const [view, setView] = useState<View>('tryon')
  const [step, setStep] = useState<Step>('upload')
  const [personImage, setPersonImage] = useState<File | null>(null)
  const [garmentImage, setGarmentImage] = useState<File | null>(null)
  const [maskImage, setMaskImage] = useState<File | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [resultId, setResultId] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS)
  const [selectedCategory, setSelectedCategory] = useState<string>('everyday')
  const [availableModels, setAvailableModels] = useState<Array<{name: string, file: string}>>([])
  const [systemStatus, setSystemStatus] = useState<'healthy' | 'degraded' | 'loading'>('loading')

  // V2 API state
  const [v2Available, setV2Available] = useState(false)
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [extractedFeatures, setExtractedFeatures] = useState<FeatureExtractionResult | null>(null)
  const [isExtractingFeatures, setIsExtractingFeatures] = useState(false)
  const [generationTime, setGenerationTime] = useState<number | null>(null)
  const [usedSeed, setUsedSeed] = useState<number | null>(null)

  // Check system health and V2 availability on load
  useEffect(() => {
    const checkHealth = async () => {
      try {
        // Check legacy API
        const response = await fetch('/api/health')
        const data = await response.json()
        setSystemStatus(data.status)
        if (data.models) {
          setAvailableModels(data.models)
        }
      } catch {
        setSystemStatus('degraded')
      }

      // Check V2 API
      const v2 = await checkV2Available()
      setV2Available(v2)

      if (v2) {
        // Load profiles if V2 is available
        const profileList = await getProfiles()
        setProfiles(profileList)

        // Get system status with hardware info
        const status = await getSystemStatus()
        if (status) {
          // Update recommended steps based on hardware
          if (status.hardware?.recommended_steps) {
            setSettings(s => ({ ...s, steps: status.hardware!.recommended_steps }))
          }
        }
      }
    }
    checkHealth()
  }, [])

  const handlePersonUpload = useCallback(async (file: File) => {
    setPersonImage(file)
    setExtractedFeatures(null)

    // Auto-extract features if V2 is available
    if (v2Available) {
      setIsExtractingFeatures(true)
      const features = await extractFeatures(file)
      if (features) {
        setExtractedFeatures(features)
      }
      setIsExtractingFeatures(false)
    }
  }, [v2Available])

  const handleGarmentUpload = useCallback((file: File) => {
    setGarmentImage(file)
  }, [])

  const handleMaskCreate = useCallback((file: File) => {
    setMaskImage(file)
    setStep('generate')
  }, [])

  const handleCategoryChange = (category: string) => {
    setSelectedCategory(category)
    const preset = PROMPT_PRESETS[category as keyof typeof PROMPT_PRESETS]
    if (preset) {
      setSettings(s => ({ ...s, prompt: preset }))
    }
  }

  const handleGenerate = async () => {
    if (!personImage || !maskImage) {
      alert('Please upload images first')
      return
    }

    setIsLoading(true)
    setProgress(0)
    setGenerationTime(null)
    setUsedSeed(null)

    // Try V2 API first if available and enabled
    if (v2Available && (settings.preserveFace || settings.preserveSkinTone)) {
      try {
        const result = await generateAdvancedTryon(personImage, {
          garmentImage: garmentImage || undefined,
          maskImage: maskImage,
          profileId: settings.selectedProfileId || undefined,
          preserve_face: settings.preserveFace,
          preserve_skin_tone: settings.preserveSkinTone,
          upscale: settings.upscale,
          face_restore: settings.faceRestore,
          steps: settings.steps,
          denoise: settings.denoise,
          prompt: settings.prompt,
          negative_prompt: settings.negativePrompt,
        })

        if (result.success && result.result_url) {
          setResultUrl(result.result_url)
          setResultId(result.result_id || null)
          setGenerationTime(result.generation_time || null)
          setUsedSeed(result.seed || null)
          setStep('result')
          setIsLoading(false)
          return
        }

        // Fall through to legacy API if V2 fails
        console.log('V2 API failed, falling back to legacy:', result.error)
      } catch (error) {
        console.error('V2 API error:', error)
      }
    }

    // Legacy API fallback
    const formData = new FormData()
    formData.append('person_image', personImage)
    formData.append('mask_image', maskImage)
    if (garmentImage) {
      formData.append('garment_image', garmentImage)
    }
    formData.append('prompt', settings.prompt)
    formData.append('negative_prompt', settings.negativePrompt)
    formData.append('steps', settings.steps.toString())
    formData.append('cfg_scale', settings.cfgScale.toString())
    formData.append('sampler', settings.sampler)
    formData.append('denoise', settings.denoise.toString())
    formData.append('model', settings.model)

    try {
      const response = await fetch('/api/tryon', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()
      setJobId(data.job_id)
      setStep('result')

      // Poll for result
      pollForResult(data.job_id)
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to start generation')
      setIsLoading(false)
    }
  }

  const pollForResult = async (id: string) => {
    let pollCount = 0
    const poll = async () => {
      try {
        const response = await fetch(`/api/tryon/${id}`)
        const data = await response.json()
        pollCount++

        setProgress(data.progress || 0)

        if (data.status === 'completed') {
          setResultUrl(data.result_url)
          setResultId(data.result_id)
          setIsLoading(false)
        } else if (data.status === 'failed') {
          alert('Generation failed: ' + (data.error || 'Unknown error'))
          setIsLoading(false)
        } else {
          // Keep polling - faster initially, slower over time
          const pollInterval = pollCount < 20 ? 1000 : pollCount < 60 ? 2000 : 3000
          setTimeout(poll, pollInterval)
        }
      } catch (error) {
        console.error('Polling error:', error)
        // Don't stop on network errors, keep trying
        setTimeout(poll, 3000)
      }
    }

    poll()
  }

  const resetAll = () => {
    setStep('upload')
    setPersonImage(null)
    setGarmentImage(null)
    setMaskImage(null)
    setResultUrl(null)
    setResultId(null)
    setJobId(null)
    setProgress(0)
    setExtractedFeatures(null)
    setGenerationTime(null)
    setUsedSeed(null)
  }

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Boutique Try-On</h1>
              <p className="text-sm text-gray-500">Professional Virtual Fitting System</p>
            </div>

            {/* Navigation */}
            <nav className="flex space-x-1">
              {[
                { id: 'tryon', label: 'Try-On' },
                { id: 'gallery', label: 'Gallery' },
                { id: 'catalog', label: 'Catalog' },
                { id: 'settings', label: 'Settings' },
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => setView(item.id as View)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                    view === item.id
                      ? 'bg-indigo-100 text-indigo-700'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </nav>

            {/* Status Indicator */}
            <div className="flex items-center space-x-3">
              {v2Available && (
                <span className="px-2 py-1 bg-green-100 text-green-700 text-xs font-medium rounded-full">
                  V2 AI
                </span>
              )}
              <div className="flex items-center space-x-2">
                <span
                  className={`w-2 h-2 rounded-full ${
                    systemStatus === 'healthy' ? 'bg-green-500' :
                    systemStatus === 'degraded' ? 'bg-yellow-500' : 'bg-gray-400'
                  }`}
                />
                <span className="text-sm text-gray-500">
                  {systemStatus === 'healthy' ? 'System Ready' :
                   systemStatus === 'degraded' ? 'ComfyUI Offline' : 'Checking...'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Try-On View */}
        {view === 'tryon' && (
          <>
            {/* Progress Steps */}
            <div className="flex justify-center mb-8">
              <div className="flex items-center space-x-4">
                {[
                  { id: 'upload', label: 'Upload' },
                  { id: 'mask', label: 'Mask' },
                  { id: 'generate', label: 'Generate' },
                  { id: 'result', label: 'Result' },
                ].map((s, i, arr) => (
                  <div key={s.id} className="flex items-center">
                    <div className="flex flex-col items-center">
                      <div
                        className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
                          step === s.id
                            ? 'bg-indigo-500 text-white'
                            : arr.findIndex(x => x.id === step) > i
                            ? 'bg-green-500 text-white'
                            : 'bg-gray-200 text-gray-600'
                        }`}
                      >
                        {i + 1}
                      </div>
                      <span className="text-xs text-gray-500 mt-1">{s.label}</span>
                    </div>
                    {i < 3 && <div className="w-12 h-0.5 bg-gray-200 mx-2 mb-4" />}
                  </div>
                ))}
              </div>
            </div>

            {/* Step Content */}
            <div className="bg-white rounded-xl shadow-lg p-6 max-w-4xl mx-auto">
              {step === 'upload' && (
                <div className="space-y-6">
                  <div className="text-center">
                    <h2 className="text-xl font-semibold">Upload Images</h2>
                    <p className="text-gray-500 mt-1">Upload a client photo and optionally a garment reference</p>
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="font-medium mb-2">Client Photo *</h3>
                      <ImageUploader
                        onUpload={handlePersonUpload}
                        accept={{ 'image/*': ['.png', '.jpg', '.jpeg'] }}
                        label="Drop photo here"
                        preview={personImage ? URL.createObjectURL(personImage) : undefined}
                      />
                      {/* Feature extraction indicator */}
                      {isExtractingFeatures && (
                        <div className="mt-2 text-sm text-indigo-600 flex items-center gap-2">
                          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          Extracting face & body features...
                        </div>
                      )}
                      {extractedFeatures && !isExtractingFeatures && (
                        <div className="mt-2 p-3 bg-green-50 rounded-lg text-sm">
                          <div className="flex items-center gap-2 text-green-700 font-medium mb-1">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                            Features Detected
                          </div>
                          <div className="text-green-600 space-y-1">
                            <div>Face: {extractedFeatures.face_detected ? 'Detected' : 'Not found'}</div>
                            <div>Body: {extractedFeatures.body_detected ? 'Detected' : 'Not found'}</div>
                            {extractedFeatures.body_shape && <div>Body type: {extractedFeatures.body_shape}</div>}
                            {extractedFeatures.pose_type && <div>Pose: {extractedFeatures.pose_type}</div>}
                          </div>
                        </div>
                      )}
                    </div>

                    <div>
                      <h3 className="font-medium mb-2">Garment Reference (Optional)</h3>
                      <ImageUploader
                        onUpload={handleGarmentUpload}
                        accept={{ 'image/*': ['.png', '.jpg', '.jpeg'] }}
                        label="Drop garment image"
                        preview={garmentImage ? URL.createObjectURL(garmentImage) : undefined}
                      />
                    </div>
                  </div>

                  <button
                    onClick={() => personImage && setStep('mask')}
                    disabled={!personImage}
                    className="w-full py-3 bg-indigo-500 text-white rounded-lg font-medium disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-indigo-600 transition"
                  >
                    Continue to Mask
                  </button>
                </div>
              )}

              {step === 'mask' && personImage && (
                <div className="space-y-6">
                  <div className="text-center">
                    <h2 className="text-xl font-semibold">Create Mask</h2>
                    <p className="text-gray-500 mt-1">Paint over the area you want to replace</p>
                  </div>

                  <MaskEditor
                    image={personImage}
                    onMaskCreate={handleMaskCreate}
                  />

                  <button
                    onClick={() => setStep('upload')}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    Back to Upload
                  </button>
                </div>
              )}

              {step === 'generate' && (
                <div className="space-y-6">
                  <div className="text-center">
                    <h2 className="text-xl font-semibold">Generation Settings</h2>
                    <p className="text-gray-500 mt-1">Configure the output</p>
                  </div>

                  {/* V2 Feature Preservation Options */}
                  {v2Available && (
                    <div className="p-4 bg-indigo-50 rounded-lg">
                      <h4 className="font-medium text-indigo-900 mb-3">AI Feature Preservation</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={settings.preserveFace}
                            onChange={(e) => setSettings(s => ({ ...s, preserveFace: e.target.checked }))}
                            className="rounded text-indigo-600"
                          />
                          <span className="text-sm">Preserve Face</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={settings.preserveSkinTone}
                            onChange={(e) => setSettings(s => ({ ...s, preserveSkinTone: e.target.checked }))}
                            className="rounded text-indigo-600"
                          />
                          <span className="text-sm">Match Skin Tone</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={settings.faceRestore}
                            onChange={(e) => setSettings(s => ({ ...s, faceRestore: e.target.checked }))}
                            className="rounded text-indigo-600"
                          />
                          <span className="text-sm">Face Restore</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={settings.upscale}
                            onChange={(e) => setSettings(s => ({ ...s, upscale: e.target.checked }))}
                            className="rounded text-indigo-600"
                          />
                          <span className="text-sm">Upscale</span>
                        </label>
                      </div>

                      {/* Profile Selection */}
                      {profiles.length > 0 && (
                        <div className="mt-4">
                          <label className="block text-sm font-medium text-indigo-900 mb-2">
                            Use Saved Profile (optional)
                          </label>
                          <select
                            value={settings.selectedProfileId}
                            onChange={(e) => setSettings(s => ({ ...s, selectedProfileId: e.target.value }))}
                            className="w-full p-2 border rounded-lg bg-white"
                          >
                            <option value="">No profile (use current image)</option>
                            {profiles.map(profile => (
                              <option key={profile.id} value={profile.id}>
                                {profile.name} - {profile.body_shape || 'Unknown shape'}
                              </option>
                            ))}
                          </select>
                          <p className="text-xs text-indigo-700 mt-1">
                            Using a saved profile ensures consistent face and body features across generations
                          </p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Category Quick Select */}
                  <div>
                    <label className="block font-medium mb-2">Garment Category</label>
                    <div className="flex flex-wrap gap-2">
                      {Object.keys(PROMPT_PRESETS).map((cat) => (
                        <button
                          key={cat}
                          onClick={() => handleCategoryChange(cat)}
                          className={`px-3 py-1 rounded-full text-sm ${
                            selectedCategory === cat
                              ? 'bg-indigo-500 text-white'
                              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                          }`}
                        >
                          {cat.charAt(0).toUpperCase() + cat.slice(1)}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Prompt */}
                  <div>
                    <label className="block font-medium mb-2">Prompt</label>
                    <textarea
                      value={settings.prompt}
                      onChange={(e) => setSettings(s => ({ ...s, prompt: e.target.value }))}
                      className="w-full p-3 border rounded-lg resize-none"
                      rows={3}
                    />
                  </div>

                  {/* Advanced Settings */}
                  <details className="border rounded-lg p-4">
                    <summary className="font-medium cursor-pointer">Advanced Settings</summary>
                    <div className="mt-4 space-y-4">
                      <div>
                        <label className="block text-sm font-medium mb-1">Negative Prompt</label>
                        <textarea
                          value={settings.negativePrompt}
                          onChange={(e) => setSettings(s => ({ ...s, negativePrompt: e.target.value }))}
                          className="w-full p-2 border rounded text-sm"
                          rows={2}
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium mb-1">Steps: {settings.steps}</label>
                          <input
                            type="range"
                            min="10"
                            max="50"
                            value={settings.steps}
                            onChange={(e) => setSettings(s => ({ ...s, steps: Number(e.target.value) }))}
                            className="w-full"
                          />
                        </div>

                        <div>
                          <label className="block text-sm font-medium mb-1">CFG: {settings.cfgScale}</label>
                          <input
                            type="range"
                            min="1"
                            max="15"
                            step="0.5"
                            value={settings.cfgScale}
                            onChange={(e) => setSettings(s => ({ ...s, cfgScale: Number(e.target.value) }))}
                            className="w-full"
                          />
                        </div>

                        <div>
                          <label className="block text-sm font-medium mb-1">Denoise: {settings.denoise}</label>
                          <input
                            type="range"
                            min="0.1"
                            max="1.0"
                            step="0.05"
                            value={settings.denoise}
                            onChange={(e) => setSettings(s => ({ ...s, denoise: Number(e.target.value) }))}
                            className="w-full"
                          />
                        </div>

                        <div>
                          <label className="block text-sm font-medium mb-1">Model</label>
                          <select
                            value={settings.model}
                            onChange={(e) => setSettings(s => ({ ...s, model: e.target.value }))}
                            className="w-full p-2 border rounded"
                          >
                            <option value="realisticVision">Realistic Vision</option>
                            <option value="sd15_inpainting">SD 1.5 Inpainting</option>
                            <option value="deliberate">Deliberate</option>
                            <option value="dreamshaper">DreamShaper</option>
                          </select>
                        </div>
                      </div>
                    </div>
                  </details>

                  <div className="text-sm text-gray-500 bg-gray-50 p-3 rounded-lg">
                    All processing is done locally. Images are stored only on this computer.
                    {v2Available && ' V2 AI engine will be used for enhanced feature preservation.'}
                  </div>

                  <button
                    onClick={handleGenerate}
                    disabled={isLoading || systemStatus !== 'healthy'}
                    className="w-full py-3 bg-green-500 text-white rounded-lg font-medium disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-green-600 transition"
                  >
                    {isLoading ? `Generating... ${progress}%` : 'Generate Try-On'}
                  </button>

                  <button
                    onClick={() => setStep('mask')}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    Back to Mask
                  </button>
                </div>
              )}

              {step === 'result' && (
                <div className="space-y-6">
                  <div className="text-center">
                    <h2 className="text-xl font-semibold">Result</h2>
                    {generationTime && (
                      <p className="text-sm text-gray-500 mt-1">
                        Generated in {generationTime.toFixed(1)}s
                        {usedSeed && ` | Seed: ${usedSeed}`}
                      </p>
                    )}
                  </div>

                  <ResultDisplay
                    originalImage={personImage ? URL.createObjectURL(personImage) : ''}
                    resultImage={resultUrl}
                    resultId={resultId}
                    jobId={jobId}
                    isLoading={isLoading}
                    progress={progress}
                  />

                  <div className="flex gap-4">
                    <button
                      onClick={resetAll}
                      className="flex-1 py-3 bg-indigo-500 text-white rounded-lg font-medium hover:bg-indigo-600 transition"
                    >
                      New Try-On
                    </button>
                    <button
                      onClick={() => setStep('generate')}
                      className="flex-1 py-3 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 transition"
                    >
                      Adjust & Regenerate
                    </button>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {/* Gallery View */}
        {view === 'gallery' && <Gallery />}

        {/* Catalog View */}
        {view === 'catalog' && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Garment Catalog</h2>
            <p className="text-gray-500">Upload and manage your garment inventory here.</p>
            {/* Catalog component would go here */}
          </div>
        )}

        {/* Settings View */}
        {view === 'settings' && <Settings models={availableModels} />}
      </div>
    </main>
  )
}
