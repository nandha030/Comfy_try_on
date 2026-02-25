'use client'

import { useRef, useState, useEffect } from 'react'

interface MaskEditorProps {
  image: File
  onMaskCreate: (maskFile: File) => void
  onSkipMask?: () => void
}

const REGIONS = [
  { id: 'top', label: 'Top', icon: '👕', description: 'Upper body clothing' },
  { id: 'bottom', label: 'Bottom', icon: '👖', description: 'Lower body clothing' },
  { id: 'dress', label: 'Dress', icon: '👗', description: 'Full-length dress' },
  { id: 'full', label: 'Full Body', icon: '🧥', description: 'All clothing areas' },
]

export default function MaskEditor({ image, onMaskCreate, onSkipMask }: MaskEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const maskCanvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [brushSize, setBrushSize] = useState(30)
  const [brushColor, setBrushColor] = useState<'white' | 'black'>('white')
  const [imageUrl, setImageUrl] = useState<string>('')
  const [isAutoMasking, setIsAutoMasking] = useState(false)
  const [autoMaskRegion, setAutoMaskRegion] = useState<string | null>(null)
  const [imgDimensions, setImgDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 })
  const [mode, setMode] = useState<'auto' | 'manual'>('auto')

  useEffect(() => {
    const url = URL.createObjectURL(image)
    setImageUrl(url)

    const img = new Image()
    img.onload = () => {
      const canvas = canvasRef.current
      const maskCanvas = maskCanvasRef.current
      if (!canvas || !maskCanvas) return

      // Set canvas size to image size (max 800px)
      const maxSize = 800
      const scale = Math.min(maxSize / img.width, maxSize / img.height, 1)
      const width = img.width * scale
      const height = img.height * scale

      canvas.width = width
      canvas.height = height
      maskCanvas.width = width
      maskCanvas.height = height
      setImgDimensions({ width, height })

      // Draw image on main canvas
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.drawImage(img, 0, 0, width, height)
      }

      // Initialize mask canvas with black (keep all)
      const maskCtx = maskCanvas.getContext('2d')
      if (maskCtx) {
        maskCtx.fillStyle = 'black'
        maskCtx.fillRect(0, 0, width, height)
      }
    }
    img.src = url

    return () => URL.revokeObjectURL(url)
  }, [image])

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || mode !== 'manual') return

    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return

    const ctx = maskCanvas.getContext('2d')
    if (!ctx) return

    const rect = maskCanvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) * (maskCanvas.width / rect.width)
    const y = (e.clientY - rect.top) * (maskCanvas.height / rect.height)

    ctx.beginPath()
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2)
    ctx.fillStyle = brushColor
    ctx.fill()
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode !== 'manual') return
    setIsDrawing(true)
    draw(e)
  }

  const handleMouseUp = () => {
    setIsDrawing(false)
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    draw(e)
  }

  const clearMask = () => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return

    const ctx = maskCanvas.getContext('2d')
    if (ctx) {
      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height)
    }
    setAutoMaskRegion(null)
  }

  const handleAutoMask = async (region: string) => {
    setIsAutoMasking(true)
    setAutoMaskRegion(region)

    try {
      const formData = new FormData()
      formData.append('person_image', image)
      formData.append('region', region)

      const response = await fetch('/api/auto-mask', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(err.detail || 'Auto-mask failed')
      }

      const blob = await response.blob()

      // Draw the auto-generated mask onto the mask canvas
      const maskImg = new Image()
      maskImg.onload = () => {
        const maskCanvas = maskCanvasRef.current
        if (!maskCanvas) return

        const ctx = maskCanvas.getContext('2d')
        if (ctx) {
          // Clear and draw the auto mask
          ctx.fillStyle = 'black'
          ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height)
          ctx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height)
        }
        URL.revokeObjectURL(maskImg.src)
        setIsAutoMasking(false)
      }
      maskImg.onerror = () => {
        URL.revokeObjectURL(maskImg.src)
        setIsAutoMasking(false)
        alert('Failed to load auto-generated mask')
      }
      maskImg.src = URL.createObjectURL(blob)
    } catch (error) {
      console.error('Auto-mask error:', error)
      setIsAutoMasking(false)
      setAutoMaskRegion(null)
      alert(`Auto-mask failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  const saveMask = () => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return

    maskCanvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], 'mask.png', { type: 'image/png' })
        onMaskCreate(file)
      }
    }, 'image/png')
  }

  return (
    <div className="space-y-4">
      {/* Mode Toggle */}
      <div className="flex items-center gap-2 p-3 bg-gray-50 rounded-lg">
        <button
          onClick={() => setMode('auto')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition ${
            mode === 'auto'
              ? 'bg-indigo-500 text-white shadow-sm'
              : 'bg-white text-gray-600 border hover:bg-gray-50'
          }`}
        >
          Auto Select
        </button>
        <button
          onClick={() => setMode('manual')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition ${
            mode === 'manual'
              ? 'bg-indigo-500 text-white shadow-sm'
              : 'bg-white text-gray-600 border hover:bg-gray-50'
          }`}
        >
          Manual Paint
        </button>
      </div>

      {/* Auto Select Region Buttons */}
      {mode === 'auto' && (
        <div className="grid grid-cols-4 gap-3">
          {REGIONS.map((region) => (
            <button
              key={region.id}
              onClick={() => handleAutoMask(region.id)}
              disabled={isAutoMasking}
              className={`flex flex-col items-center gap-1 p-3 rounded-lg border-2 transition ${
                autoMaskRegion === region.id
                  ? 'border-indigo-500 bg-indigo-50'
                  : 'border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/50'
              } ${isAutoMasking ? 'opacity-60 cursor-wait' : 'cursor-pointer'}`}
            >
              <span className="text-2xl">{region.icon}</span>
              <span className="text-sm font-medium">{region.label}</span>
              <span className="text-xs text-gray-500">{region.description}</span>
            </button>
          ))}
        </div>
      )}

      {/* Auto-masking loading indicator */}
      {isAutoMasking && (
        <div className="flex items-center justify-center gap-2 py-2 text-indigo-600">
          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <span className="text-sm font-medium">Detecting clothing region...</span>
        </div>
      )}

      {/* Manual Paint Toolbar */}
      {mode === 'manual' && (
        <div className="flex flex-wrap items-center gap-4 p-3 bg-gray-100 rounded-lg">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">Brush:</label>
            <button
              onClick={() => setBrushColor('white')}
              className={`px-3 py-1 rounded text-sm ${
                brushColor === 'white'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white border'
              }`}
            >
              Paint (Replace)
            </button>
            <button
              onClick={() => setBrushColor('black')}
              className={`px-3 py-1 rounded text-sm ${
                brushColor === 'black'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white border'
              }`}
            >
              Erase (Keep)
            </button>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">Size:</label>
            <input
              type="range"
              min="5"
              max="100"
              value={brushSize}
              onChange={(e) => setBrushSize(Number(e.target.value))}
              className="w-24"
            />
            <span className="text-sm w-8">{brushSize}</span>
          </div>

          <button
            onClick={clearMask}
            className="px-3 py-1 bg-red-100 text-red-600 rounded text-sm hover:bg-red-200"
          >
            Clear
          </button>
        </div>
      )}

      {/* Canvas Container */}
      <div className="relative inline-block border rounded-lg overflow-hidden">
        {/* Background Image */}
        <canvas ref={canvasRef} className="block" />

        {/* Mask Overlay */}
        <canvas
          ref={maskCanvasRef}
          className={`absolute top-0 left-0 opacity-50 ${
            mode === 'manual' ? 'cursor-crosshair' : 'cursor-default'
          }`}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseUp}
        />
      </div>

      <p className="text-sm text-gray-500">
        {mode === 'auto'
          ? 'Select a clothing region above to auto-detect the area. You can switch to Manual Paint to refine.'
          : 'Paint white over the area you want to replace. Switch to Auto Select for quick region detection.'}
      </p>

      <div className="flex gap-3">
        <button
          onClick={clearMask}
          className="px-4 py-3 border border-gray-300 text-gray-600 rounded-lg font-medium hover:bg-gray-50 transition"
        >
          Reset Mask
        </button>
        {onSkipMask && (
          <button
            onClick={onSkipMask}
            className="px-4 py-3 border border-amber-300 text-amber-700 bg-amber-50 rounded-lg font-medium hover:bg-amber-100 transition text-sm"
          >
            Skip (Auto-detect)
          </button>
        )}
        <button
          onClick={saveMask}
          className="flex-1 py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition"
        >
          Continue with this Mask
        </button>
      </div>
    </div>
  )
}
