import React from 'react';
import { History } from 'lucide-react';

export default function HistoryPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Detection History</h1>
        <p className="text-gray-600">Browse, search, and manage your detection history</p>
      </div>
      
      <div className="bg-white rounded-lg shadow p-8">
        <div className="text-center">
          <History className="mx-auto h-16 w-16 text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900">History Page - Coming Soon</h3>
          <p className="text-gray-600 mt-2">
            This page will show your detection history with search and export capabilities.
          </p>
        </div>
      </div>
    </div>
  );
}