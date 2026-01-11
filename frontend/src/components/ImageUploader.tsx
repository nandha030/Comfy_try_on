'use client'

import { useCallback } from 'react'
import { useDropzone, Accept } from 'react-dropzone'

interface ImageUploaderProps {
  onUpload: (file: File) => void
  accept?: Accept
  label?: string
  preview?: string
}

export default function ImageUploader({
  onUpload,
  accept = { 'image/*': ['.png', '.jpg', '.jpeg'] },
  label = 'Drop image here or click to upload',
  preview,
}: ImageUploaderProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onUpload(acceptedFiles[0])
      }
    },
    [onUpload]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles: 1,
  })

  return (
    <div
      {...getRootProps()}
      className={`dropzone ${isDragActive ? 'active' : ''} ${
        preview ? 'p-2' : ''
      }`}
    >
      <input {...getInputProps()} />
      {preview ? (
        <div className="relative">
          <img
            src={preview}
            alt="Preview"
            className="max-h-64 mx-auto rounded"
          />
          <p className="text-sm text-gray-500 mt-2">Click to change</p>
        </div>
      ) : (
        <div className="py-8">
          <svg
            className="w-12 h-12 mx-auto text-gray-400 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <p className="text-gray-600">{label}</p>
          <p className="text-sm text-gray-400 mt-1">PNG, JPG up to 10MB</p>
        </div>
      )}
    </div>
  )
}
