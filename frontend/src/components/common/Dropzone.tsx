import React, { useRef, useState } from 'react';
import { UploadCloud } from 'lucide-react';

interface DropzoneProps {
  onImageSelected: (imageUrl: string, file?: File, filename?: string) => void;
  isLoading?: boolean;
}

export const Dropzone: React.FC<DropzoneProps> = ({ onImageSelected, isLoading = false }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    if (!file.type.match(/^image\/(jpeg|jpg|png)$/i)) {
      alert('Please upload a valid endoscopy image file (.jpg or .png)');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        onImageSelected(e.target.result as string, file, file.name);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
      className={`w-full rounded-upload border-2 border-dashed p-7 flex flex-col items-center justify-center cursor-pointer transition-all duration-200 group relative overflow-hidden ${
        isDragOver
          ? 'border-brand-500 bg-gradient-to-b from-brand-50/70 to-indigo-50/50 scale-[1.01] shadow-md shadow-brand-500/10'
          : 'border-slate-300/90 hover:border-brand-500 bg-gradient-to-b from-slate-50/80 via-white to-slate-50/50 hover:from-brand-50/40 hover:via-white hover:to-indigo-50/30 hover:shadow-sm'
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png, image/jpeg, image/jpg"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
          }
        }}
      />

      <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-brand-50 to-indigo-50 border border-brand-200/80 shadow-sm flex items-center justify-center mb-3 group-hover:scale-110 group-hover:shadow-md group-hover:shadow-brand-500/20 group-hover:border-brand-400 transition-all duration-200">
        <UploadCloud className="w-7 h-7 text-brand-600 group-hover:text-brand-700 transition-colors" />
      </div>

      <p className="text-sm font-bold text-slate-800 tracking-tight group-hover:text-brand-700 transition-colors">
        Click or Drag & Drop Endoscopy Image
      </p>

      <span className="text-xs font-medium text-slate-400 mt-1">
        (JPG, PNG) • Resolution auto-normalized to 256×256
      </span>

      {isLoading && (
        <div className="mt-3 px-3.5 py-1.5 rounded-full bg-gradient-to-r from-brand-600 to-indigo-600 text-white text-xs font-semibold flex items-center gap-2 shadow-md shadow-brand-500/25 animate-pulse">
          <span className="w-2 h-2 rounded-full bg-white animate-ping" />
          <span>Processing PyTorch CUDA Tensor...</span>
        </div>
      )}
    </div>
  );
};
