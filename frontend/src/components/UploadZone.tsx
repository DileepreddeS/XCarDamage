// UploadZone.tsx
// Drag and drop image upload area

import { useState, useCallback } from 'react';

interface Props {
  onUpload: (file: File) => void;
  loading: boolean;
}

export default function UploadZone({ onUpload, loading }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);
    // Send to parent
    onUpload(file);
  }, [onUpload]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="w-full">
      <label
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        className={`
          relative flex flex-col items-center justify-center
          w-full h-64 rounded-2xl border-2 border-dashed cursor-pointer
          ${dragOver
            ? 'border-blue-400 bg-blue-500/10'
            : 'border-gray-700 bg-gray-900/50 hover:border-gray-500 hover:bg-gray-900'
          }
        `}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleChange}
          className="hidden"
          disabled={loading}
        />

        {loading ? (
          /* Loading state */
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-gray-400 text-sm">Analyzing damage...</p>
          </div>
        ) : preview ? (
          /* Preview state */
          <div className="flex flex-col items-center gap-3">
            <img src={preview} alt="Preview" className="h-40 w-auto rounded-lg object-contain" />
            <p className="text-gray-400 text-sm">Click or drag to analyze a different image</p>
          </div>
        ) : (
          /* Default state */
          <div className="flex flex-col items-center gap-3">
            <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center text-3xl">
              🚗
            </div>
            <div className="text-center">
              <p className="text-white font-semibold">Drop a car image here</p>
              <p className="text-gray-400 text-sm mt-1">or click to browse</p>
            </div>
            <p className="text-gray-600 text-xs">PNG, JPG, JPEG supported</p>
          </div>
        )}
      </label>
    </div>
  )
}