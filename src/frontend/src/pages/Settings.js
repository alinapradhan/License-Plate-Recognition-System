import React from 'react';
import { Settings } from 'lucide-react';

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600">Configure system settings and privacy options</p>
      </div>
      
      <div className="bg-white rounded-lg shadow p-8">
        <div className="text-center">
          <Settings className="mx-auto h-16 w-16 text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900">Settings Page - Coming Soon</h3>
          <p className="text-gray-600 mt-2">
            This page will provide system configuration and privacy settings.
          </p>
        </div>
      </div>
    </div>
  );
}