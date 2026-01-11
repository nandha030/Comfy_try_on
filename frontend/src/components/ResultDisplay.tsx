'use client'

import { useState } from 'react'

interface ResultDisplayProps {
  originalImage: string
  resultImage: string | null
  resultId?: string | null
  jobId: string | null
  isLoading?: boolean
  progress?: number
}

export default function ResultDisplay({
  originalImage,
  resultImage,
  resultId,
  jobId,
  isLoading = false,
  progress = 0,
}: ResultDisplayProps) {
  const [sliderPosition, setSliderPosition] = useState(50)
  const [isFavorite, setIsFavorite] = useState(false)

  const toggleFavorite = async () => {
    if (!resultId) return

    try {
      const formData = new FormData()
      formData.append('is_favorite', (!isFavorite).toString())

      await fetch(`/api/results/${resultId}/favorite`, {
        method: 'POST',
        body: formData
      })

      setIsFavorite(!isFavorite)
    } catch (error) {
      console.error('Failed to toggle favorite:', error)
    }
  }

  if (isLoading || !resultImage) {
    return (
      <div className="text-center py-12">
        <div className="relative w-24 h-24 mx-auto mb-4">
          {/* Circular progress */}
          <svg className="w-24 h-24 transform -rotate-90">
            <circle
              cx="48"
              cy="48"
              r="40"
              stroke="#e5e7eb"
              strokeWidth="8"
              fill="none"
            />
            <circle
              cx="48"
              cy="48"
              r="40"
              stroke="#6366f1"
              strokeWidth="8"
              fill="none"
              strokeLinecap="round"
              strokeDasharray={251.2}
              strokeDashoffset={251.2 - (251.2 * progress) / 100}
              className="transition-all duration-300"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-lg font-semibold text-indigo-600">{progress}%</span>
          </div>
        </div>
        <p className="text-gray-600 font-medium">Generating your try-on...</p>
        <p className="text-sm text-gray-400 mt-2">
          This may take a few minutes on CPU
        </p>
        {jobId && (
          <p className="text-xs text-gray-400 mt-4 font-mono">Job: {jobId}</p>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Compare Slider */}
      <div className="relative w-full max-w-2xl mx-auto overflow-hidden rounded-lg shadow-lg">
        {/* Result Image (Background) */}
        <img
          src={resultImage}
          alt="Result"
          className="w-full block"
        />

        {/* Original Image (Overlay with clip) */}
        <div
          className="absolute top-0 left-0 h-full overflow-hidden"
          style={{ width: `${sliderPosition}%` }}
        >
          <img
            src={originalImage}
            alt="Original"
            className="h-full object-cover"
            style={{ width: `${100 / (sliderPosition / 100)}%`, maxWidth: 'none' }}
          />
        </div>

        {/* Slider Line */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-white shadow-lg cursor-ew-resize"
          style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 bg-white rounded-full shadow-lg flex items-center justify-center border-2 border-gray-200">
            <svg
              className="w-5 h-5 text-gray-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 9l4-4 4 4m0 6l-4 4-4-4"
              />
            </svg>
          </div>
        </div>

        {/* Invisible slider input */}
        <input
          type="range"
          min="0"
          max="100"
          value={sliderPosition}
          onChange={(e) => setSliderPosition(Number(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize"
        />

        {/* Labels */}
        <div className="absolute bottom-4 left-4 px-2 py-1 bg-black bg-opacity-50 text-white text-xs rounded">
          Original
        </div>
        <div className="absolute bottom-4 right-4 px-2 py-1 bg-black bg-opacity-50 text-white text-xs rounded">
          Result
        </div>
      </div>

      {/* Slider hint */}
      <p className="text-center text-sm text-gray-400">
        Drag the slider to compare original and result
      </p>

      {/* Actions */}
      <div className="flex justify-center gap-3">
        <button
          onClick={toggleFavorite}
          className={`px-4 py-2 rounded-lg font-medium transition flex items-center gap-2 ${
            isFavorite
              ? 'bg-red-100 text-red-600 hover:bg-red-200'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          <svg
            className="w-5 h-5"
            fill={isFavorite ? 'currentColor' : 'none'}
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
            />
          </svg>
          {isFavorite ? 'Favorited' : 'Favorite'}
        </button>

        <a
          href={resultImage}
          download={`tryon-${resultId || 'result'}.png`}
          className="px-4 py-2 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
          Download
        </a>
      </div>
    </div>
  )
}
