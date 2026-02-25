'use client'

import { useState, useEffect } from 'react'

interface Result {
  id: string
  image_path: string
  is_favorite: boolean
  created_at: string
  seed?: number
  generation_time?: number
  model_used?: string
  session?: {
    prompt?: string
  }
}

export default function Gallery() {
  const [results, setResults] = useState<Result[]>([])
  const [favorites, setFavorites] = useState<Result[]>([])
  const [filter, setFilter] = useState<'all' | 'favorites'>('all')
  const [selectedImage, setSelectedImage] = useState<Result | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    fetchResults()
  }, [])

  const fetchResults = async () => {
    setIsLoading(true)
    try {
      const [resultsRes, favoritesRes] = await Promise.all([
        fetch('/api/results?limit=100'),
        fetch('/api/results/favorites')
      ])

      const resultsData = await resultsRes.json()
      const favoritesData = await favoritesRes.json()

      setResults(resultsData.results || [])
      setFavorites(favoritesData.favorites || [])
    } catch (error) {
      console.error('Failed to fetch results:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const toggleFavorite = async (resultId: string, currentState: boolean) => {
    try {
      const formData = new FormData()
      formData.append('is_favorite', (!currentState).toString())

      await fetch(`/api/results/${resultId}/favorite`, {
        method: 'POST',
        body: formData
      })

      // Refresh data
      fetchResults()
    } catch (error) {
      console.error('Failed to toggle favorite:', error)
    }
  }

  const displayedResults = filter === 'favorites' ? favorites : results

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Gallery</h2>
          <p className="text-gray-500">Browse your generated try-on results</p>
        </div>

        <div className="flex items-center gap-4">
          {/* Filter Tabs */}
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setFilter('all')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition ${
                filter === 'all' ? 'bg-white shadow' : 'text-gray-600'
              }`}
            >
              All ({results.length})
            </button>
            <button
              onClick={() => setFilter('favorites')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition ${
                filter === 'favorites' ? 'bg-white shadow' : 'text-gray-600'
              }`}
            >
              Favorites ({favorites.length})
            </button>
          </div>

          <button
            onClick={fetchResults}
            className="p-2 text-gray-500 hover:text-gray-700 rounded-lg hover:bg-gray-100"
            title="Refresh"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="text-center py-12">
          <div className="animate-spin w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full mx-auto" />
          <p className="mt-2 text-gray-500">Loading...</p>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && displayedResults.length === 0 && (
        <div className="text-center py-12 bg-white rounded-xl">
          <svg className="w-16 h-16 mx-auto text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <h3 className="mt-4 text-lg font-medium text-gray-900">No results yet</h3>
          <p className="mt-1 text-gray-500">
            {filter === 'favorites'
              ? 'Mark some results as favorites to see them here'
              : 'Start a try-on session to generate results'}
          </p>
        </div>
      )}

      {/* Image Grid */}
      {!isLoading && displayedResults.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {displayedResults.map((result) => (
            <div
              key={result.id}
              className="relative group bg-white rounded-lg overflow-hidden shadow-sm hover:shadow-md transition cursor-pointer"
              onClick={() => setSelectedImage(result)}
            >
              <img
                src={`/api/results/${result.id}/image`}
                alt="Try-on result"
                className="w-full aspect-[3/4] object-cover"
              />

              {/* Overlay */}
              <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition flex items-end justify-between p-2">
                <span className="text-white text-xs opacity-0 group-hover:opacity-100 transition">
                  {formatDate(result.created_at)}
                </span>

                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    toggleFavorite(result.id, result.is_favorite)
                  }}
                  className={`p-1 rounded-full transition ${
                    result.is_favorite
                      ? 'text-red-500 bg-white'
                      : 'text-white opacity-0 group-hover:opacity-100'
                  }`}
                >
                  <svg className="w-5 h-5" fill={result.is_favorite ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                  </svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div
            className="bg-white rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex">
              {/* Image */}
              <div className="flex-1 bg-gray-100">
                <img
                  src={`/api/results/${selectedImage.id}/image`}
                  alt="Try-on result"
                  className="w-full h-full object-contain max-h-[80vh]"
                />
              </div>

              {/* Details */}
              <div className="w-72 p-4 border-l flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Details</h3>
                  <button
                    onClick={() => setSelectedImage(null)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                <div className="space-y-3 text-sm flex-1">
                  <div>
                    <span className="text-gray-500">Created</span>
                    <p>{formatDate(selectedImage.created_at)}</p>
                  </div>

                  {selectedImage.model_used && (
                    <div>
                      <span className="text-gray-500">Model</span>
                      <p>{selectedImage.model_used}</p>
                    </div>
                  )}

                  {selectedImage.seed && (
                    <div>
                      <span className="text-gray-500">Seed</span>
                      <p className="font-mono text-xs">{selectedImage.seed}</p>
                    </div>
                  )}

                  {selectedImage.generation_time && (
                    <div>
                      <span className="text-gray-500">Generation Time</span>
                      <p>{selectedImage.generation_time.toFixed(1)}s</p>
                    </div>
                  )}

                  {selectedImage.session?.prompt && (
                    <div>
                      <span className="text-gray-500">Prompt</span>
                      <p className="text-xs mt-1">{selectedImage.session.prompt}</p>
                    </div>
                  )}
                </div>

                <div className="pt-4 border-t space-y-2">
                  <button
                    onClick={() => toggleFavorite(selectedImage.id, selectedImage.is_favorite)}
                    className={`w-full py-2 rounded-lg font-medium transition ${
                      selectedImage.is_favorite
                        ? 'bg-red-100 text-red-600 hover:bg-red-200'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {selectedImage.is_favorite ? 'Remove from Favorites' : 'Add to Favorites'}
                  </button>

                  <a
                    href={`/api/results/${selectedImage.id}/image`}
                    download={`tryon-${selectedImage.id}.png`}
                    className="block w-full py-2 bg-indigo-500 text-white text-center rounded-lg font-medium hover:bg-indigo-600 transition"
                  >
                    Download
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
