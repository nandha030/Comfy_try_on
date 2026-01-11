'use client'

import { useState, useEffect, useRef } from 'react'
import {
  getSystemStatus,
  detectHardware,
  getProfiles,
  createProfile,
  deleteProfile,
  type SystemStatus,
  type HardwareProfile,
  type ModelProfile,
} from '@/lib/api'

interface SettingsProps {
  models: Array<{ name: string; file: string }>
}

interface Stats {
  clients: number
  garments: number
  sessions: number
  results: number
  favorites: number
}

export default function Settings({ models }: SettingsProps) {
  const [stats, setStats] = useState<Stats | null>(null)
  const [isClearing, setIsClearing] = useState(false)
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [hardware, setHardware] = useState<HardwareProfile | null>(null)
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [isDetectingHardware, setIsDetectingHardware] = useState(false)
  const [isLoadingProfiles, setIsLoadingProfiles] = useState(true)
  const [isCreatingProfile, setIsCreatingProfile] = useState(false)
  const [newProfileName, setNewProfileName] = useState('')
  const [showCreateProfile, setShowCreateProfile] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchStats()
    fetchSystemStatus()
    fetchProfiles()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/data/stats')
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  const fetchSystemStatus = async () => {
    const status = await getSystemStatus()
    setSystemStatus(status)
    if (status?.hardware) {
      setHardware(status.hardware)
    }
  }

  const fetchProfiles = async () => {
    setIsLoadingProfiles(true)
    const profileList = await getProfiles()
    setProfiles(profileList)
    setIsLoadingProfiles(false)
  }

  const handleDetectHardware = async () => {
    setIsDetectingHardware(true)
    const hw = await detectHardware()
    if (hw) {
      setHardware(hw)
    }
    setIsDetectingHardware(false)
  }

  const handleCreateProfile = async (file: File) => {
    if (!newProfileName.trim()) {
      alert('Please enter a profile name')
      return
    }

    setIsCreatingProfile(true)
    const profile = await createProfile(file, newProfileName.trim())
    if (profile) {
      setProfiles([...profiles, profile])
      setNewProfileName('')
      setShowCreateProfile(false)
    } else {
      alert('Failed to create profile')
    }
    setIsCreatingProfile(false)
  }

  const handleDeleteProfile = async (profileId: string) => {
    if (!confirm('Are you sure you want to delete this profile?')) return

    const success = await deleteProfile(profileId)
    if (success) {
      setProfiles(profiles.filter((p) => p.id !== profileId))
    } else {
      alert('Failed to delete profile')
    }
  }

  const clearResults = async () => {
    if (!confirm('Are you sure you want to clear all generated results? This cannot be undone.')) {
      return
    }

    setIsClearing(true)
    try {
      await fetch('/api/data/clear-results', { method: 'DELETE' })
      await fetch('/api/data/clear-uploads', { method: 'DELETE' })
      fetchStats()
      alert('Results cleared successfully')
    } catch (error) {
      console.error('Failed to clear results:', error)
      alert('Failed to clear results')
    } finally {
      setIsClearing(false)
    }
  }

  const getDeviceIcon = (deviceType: string) => {
    switch (deviceType) {
      case 'cuda':
        return '🟢'
      case 'mps':
        return '🍎'
      case 'directml':
        return '🔷'
      default:
        return '💻'
    }
  }

  const getSpeedColor = (speed: string) => {
    if (speed.includes('fast')) return 'text-green-600'
    if (speed.includes('moderate')) return 'text-yellow-600'
    return 'text-orange-600'
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h2 className="text-2xl font-bold">Settings</h2>
        <p className="text-gray-500">Manage your system configuration</p>
      </div>

      {/* Hardware Information */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Hardware Configuration</h3>
          <button
            onClick={handleDetectHardware}
            disabled={isDetectingHardware}
            className="px-3 py-1 text-sm bg-indigo-100 text-indigo-700 rounded-lg hover:bg-indigo-200 disabled:opacity-50 transition"
          >
            {isDetectingHardware ? 'Detecting...' : 'Re-detect Hardware'}
          </button>
        </div>

        {hardware ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className="text-2xl mb-1">{getDeviceIcon(hardware.device_type)}</div>
                <div className="font-medium text-sm">{hardware.device_name}</div>
                <div className="text-xs text-gray-500">{hardware.device_type.toUpperCase()}</div>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-indigo-600">
                  {hardware.vram_gb > 0 ? `${hardware.vram_gb}GB` : 'N/A'}
                </div>
                <div className="text-xs text-gray-500">VRAM</div>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-indigo-600">{hardware.recommended_steps}</div>
                <div className="text-xs text-gray-500">Rec. Steps</div>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className={`text-lg font-medium ${getSpeedColor(hardware.estimated_speed)}`}>
                  {hardware.estimated_speed}
                </div>
                <div className="text-xs text-gray-500">Speed</div>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">
                {hardware.system}
              </span>
              <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">
                {hardware.arch}
              </span>
              <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                {hardware.compute_backend}
              </span>
              <span className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-xs font-medium">
                Profile: {hardware.model_profile}
              </span>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>Hardware information not available</p>
            <button
              onClick={handleDetectHardware}
              disabled={isDetectingHardware}
              className="mt-2 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 disabled:opacity-50 transition"
            >
              {isDetectingHardware ? 'Detecting...' : 'Detect Hardware'}
            </button>
          </div>
        )}
      </div>

      {/* Model Profiles */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">Model Profiles</h3>
            <p className="text-sm text-gray-500">
              Save face & body features for consistent results
            </p>
          </div>
          <button
            onClick={() => setShowCreateProfile(true)}
            className="px-3 py-1 text-sm bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition"
          >
            + New Profile
          </button>
        </div>

        {showCreateProfile && (
          <div className="mb-4 p-4 bg-gray-50 rounded-lg">
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium mb-1">Profile Name</label>
                <input
                  type="text"
                  value={newProfileName}
                  onChange={(e) => setNewProfileName(e.target.value)}
                  placeholder="e.g., Client A, Model 1"
                  className="w-full p-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Reference Image</label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleCreateProfile(file)
                  }}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isCreatingProfile || !newProfileName.trim()}
                  className="w-full p-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-500 hover:border-indigo-400 hover:text-indigo-500 disabled:opacity-50 transition"
                >
                  {isCreatingProfile ? 'Creating profile...' : 'Click to upload reference photo'}
                </button>
              </div>
              <button
                onClick={() => setShowCreateProfile(false)}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {isLoadingProfiles ? (
          <div className="text-center py-8 text-gray-500">Loading profiles...</div>
        ) : profiles.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <p>No profiles created yet</p>
            <p className="text-sm mt-1">
              Create a profile to save face and body features for consistent try-on results
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {profiles.map((profile) => (
              <div
                key={profile.id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600 font-semibold">
                    {profile.name.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <div className="font-medium">{profile.name}</div>
                    <div className="text-xs text-gray-500">
                      {profile.face_embedding ? 'Face detected' : 'No face data'} |{' '}
                      {profile.body_shape || 'Unknown shape'}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400">
                    {new Date(profile.created_at).toLocaleDateString()}
                  </span>
                  <button
                    onClick={() => handleDeleteProfile(profile.id)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* System Status */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">System Status</h3>

        <div className="space-y-4">
          <div className="flex items-center justify-between py-3 border-b">
            <div>
              <span className="font-medium">AI Engine (V2)</span>
              <p className="text-sm text-gray-500">Advanced face & body preservation</p>
            </div>
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                systemStatus?.ai_engine
                  ? 'bg-green-100 text-green-700'
                  : 'bg-gray-100 text-gray-700'
              }`}
            >
              {systemStatus?.ai_engine ? 'Active' : 'Not Available'}
            </span>
          </div>

          <div className="flex items-center justify-between py-3 border-b">
            <div>
              <span className="font-medium">ComfyUI</span>
              <p className="text-sm text-gray-500">Image generation backend</p>
            </div>
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                systemStatus?.comfyui
                  ? 'bg-green-100 text-green-700'
                  : 'bg-yellow-100 text-yellow-700'
              }`}
            >
              {systemStatus?.comfyui ? 'Connected' : 'Offline'}
            </span>
          </div>

          <div className="flex items-center justify-between py-3 border-b">
            <div>
              <span className="font-medium">Mode</span>
              <p className="text-sm text-gray-500">All data stored locally</p>
            </div>
            <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
              Local Only
            </span>
          </div>

          <div className="flex items-center justify-between py-3">
            <div>
              <span className="font-medium">Content Restrictions</span>
              <p className="text-sm text-gray-500">Professional boutique use</p>
            </div>
            <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
              Unrestricted
            </span>
          </div>
        </div>
      </div>

      {/* Available Models */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Available Models</h3>

        {models.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <p>No models found. Run the installer:</p>
            <code className="block mt-2 bg-gray-100 p-2 rounded text-sm">python install.py</code>
          </div>
        ) : (
          <div className="space-y-2">
            {models.map((model) => (
              <div
                key={model.file}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div>
                  <span className="font-medium">{model.name}</span>
                  <p className="text-sm text-gray-500">{model.file}</p>
                </div>
                <span className="text-green-500">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                </span>
              </div>
            ))}
          </div>
        )}

        <div className="mt-4 p-4 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-800">
            <strong>Tip:</strong> For best results with fashion photography, use Realistic Vision
            or ChilloutMix models.
          </p>
        </div>
      </div>

      {/* Statistics */}
      {stats && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Statistics</h3>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-indigo-600">{stats.clients}</div>
              <div className="text-sm text-gray-500">Clients</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-indigo-600">{stats.garments}</div>
              <div className="text-sm text-gray-500">Garments</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-indigo-600">{stats.sessions}</div>
              <div className="text-sm text-gray-500">Sessions</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-indigo-600">{stats.results}</div>
              <div className="text-sm text-gray-500">Results</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-red-500">{stats.favorites}</div>
              <div className="text-sm text-gray-500">Favorites</div>
            </div>
          </div>
        </div>
      )}

      {/* Data Management */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Data Management</h3>

        <div className="space-y-4">
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <span className="font-medium">Clear Generated Results</span>
                <p className="text-sm text-gray-500">
                  Remove all try-on results and temporary uploads
                </p>
              </div>
              <button
                onClick={clearResults}
                disabled={isClearing}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 transition"
              >
                {isClearing ? 'Clearing...' : 'Clear Results'}
              </button>
            </div>
          </div>

          <div className="p-4 bg-yellow-50 rounded-lg">
            <p className="text-sm text-yellow-800">
              <strong>Note:</strong> Clearing results will not affect your garment catalog or client
              data. This action cannot be undone.
            </p>
          </div>
        </div>
      </div>

      {/* File Locations */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">File Locations</h3>

        <div className="space-y-3 text-sm">
          <div className="flex justify-between p-3 bg-gray-50 rounded">
            <span className="text-gray-600">Database</span>
            <code className="text-xs">backend/data/boutique.db</code>
          </div>
          <div className="flex justify-between p-3 bg-gray-50 rounded">
            <span className="text-gray-600">Results</span>
            <code className="text-xs">backend/data/results/</code>
          </div>
          <div className="flex justify-between p-3 bg-gray-50 rounded">
            <span className="text-gray-600">Garments</span>
            <code className="text-xs">backend/data/garments/</code>
          </div>
          <div className="flex justify-between p-3 bg-gray-50 rounded">
            <span className="text-gray-600">Model Profiles</span>
            <code className="text-xs">models/data/profiles/</code>
          </div>
          <div className="flex justify-between p-3 bg-gray-50 rounded">
            <span className="text-gray-600">AI Models</span>
            <code className="text-xs">models/</code>
          </div>
        </div>
      </div>
    </div>
  )
}
