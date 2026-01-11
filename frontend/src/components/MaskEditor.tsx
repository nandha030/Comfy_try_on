'use client'

import { useRef, useState, useEffect } from 'react'

interface MaskEditorProps {
  image: File
  onMaskCreate: (maskFile: File) => void
}

export default function MaskEditor({ image, onMaskCreate }: MaskEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const maskCanvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [brushSize, setBrushSize] = useState(30)
  const [brushColor, setBrushColor] = useState<'white' | 'black'>('white')
  const [imageUrl, setImageUrl] = useState<string>('')

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
    if (!isDrawing) return

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
      {/* Toolbar */}
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

      {/* Canvas Container */}
      <div className="relative inline-block border rounded-lg overflow-hidden">
        {/* Background Image */}
        <canvas ref={canvasRef} className="block" />

        {/* Mask Overlay */}
        <canvas
          ref={maskCanvasRef}
          className="absolute top-0 left-0 opacity-50 cursor-crosshair"
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseUp}
        />
      </div>

      <p className="text-sm text-gray-500">
        Paint white over the clothing area you want to replace. The mask will be
        used to guide the AI.
      </p>

      <button
        onClick={saveMask}
        className="w-full py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition"
      >
        Continue with this Mask
      </button>
    </div>
  )
}
