import React from 'react';
import { Upload } from 'lucide-react';

export default function UploadPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Upload & Process</h1>
        <p className="text-gray-600">Upload images for license plate detection and recognition</p>
      </div>
      
      <div className="bg-white rounded-lg shadow p-8">
        <div className="text-center">
          <Upload className="mx-auto h-16 w-16 text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900">Upload Page - Coming Soon</h3>
          <p className="text-gray-600 mt-2">
            This page will allow you to upload and process images for license plate detection.
          </p>
        </div>
      </div>
    </div>
  );
}