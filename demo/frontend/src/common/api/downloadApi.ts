/**
 * API interaction functions for downloading data from the backend.
 */
import { getFileName } from '@/common/components/options/ShareUtils'; // Reuse filename helper
import { sessionAtom } from '@/demo/atoms'; // Atom to get current session
import { useAtomValue } from 'jotai';
import Logger from '@/common/logger/Logger';

// Utility to get base API URL (adjust if needed)
const getApiBaseUrl = (): string => {
  // Example: read from environment variable or settings
  // Replace with your actual API endpoint logic
  const endpoint = window.APP_SETTINGS?.videoAPIEndpoint ?? '/api'; // Use /api as default relative path
  return endpoint.replace('/graphql', ''); // Remove /graphql if present
};


// Helper to trigger blob download
const triggerBlobDownload = (blob: Blob, defaultFilename: string) => {
  try {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style.display = 'none';
    a.href = url;
    // Use a specific filename based on type
    a.download = defaultFilename;
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
  } catch (error) {
     Logger.error('Error triggering blob download:', error);
     // Consider showing an error message to the user
     alert(`Failed to trigger download for ${defaultFilename}. See console for details.`);
  }
};

// --- API Call Functions ---

export const downloadMasksJson = async (sessionId: string): Promise<void> => {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}/sessions/${sessionId}/download_masks`; // Assuming GET endpoint
  Logger.info(`Downloading masks JSON from: ${url}`);
  try {
    const response = await fetch(url, { method: 'GET' }); // Use GET
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
      throw new Error(`HTTP error ${response.status}: ${errorData.error || response.statusText}`);
    }
    const jsonData = await response.json();
    const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
    triggerBlobDownload(blob, `${sessionId}_masks.json`);
  } catch (error) {
    Logger.error(`Failed to download masks for session ${sessionId}:`, error);
    alert(`Failed to download masks: ${error instanceof Error ? error.message : String(error)}`);
    throw error; // Re-throw for loading state handling
  }
};

export const downloadImagesZip = async (sessionId: string): Promise<void> => {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}/sessions/${sessionId}/download_images_zip`;
  Logger.info(`Downloading images zip from: ${url}`);
  try {
    const response = await fetch(url);
     if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
      throw new Error(`HTTP error ${response.status}: ${errorData.error || response.statusText}`);
    }
    const blob = await response.blob(); // Get blob directly
    triggerBlobDownload(blob, `${sessionId}_images.zip`);
  } catch (error) {
    Logger.error(`Failed to download images zip for session ${sessionId}:`, error);
    alert(`Failed to download images: ${error instanceof Error ? error.message : String(error)}`);
    throw error;
  }
};

export const downloadYoloLabelsZip = async (sessionId: string): Promise<void> => {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}/sessions/${sessionId}/download_yolo_labels`;
  Logger.info(`Downloading YOLO labels zip from: ${url}`);
   try {
    const response = await fetch(url);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
      throw new Error(`HTTP error ${response.status}: ${errorData.error || response.statusText}`);
    }
    const blob = await response.blob();
    triggerBlobDownload(blob, `${sessionId}_yolo_labels.zip`);
  } catch (error) {
    Logger.error(`Failed to download YOLO labels zip for session ${sessionId}:`, error);
    alert(`Failed to download YOLO labels: ${error instanceof Error ? error.message : String(error)}`);
    throw error;
  }
};

export const downloadYoloFormatZip = async (sessionId: string): Promise<void> => {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}/sessions/${sessionId}/download_yolo_format`;
  Logger.info(`Downloading YOLO format zip from: ${url}`);
   try {
    const response = await fetch(url);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
      throw new Error(`HTTP error ${response.status}: ${errorData.error || response.statusText}`);
    }
    const blob = await response.blob();
    triggerBlobDownload(blob, `${sessionId}_yolo_dataset.zip`);
  } catch (error) {
    Logger.error(`Failed to download YOLO format zip for session ${sessionId}:`, error);
    alert(`Failed to download YOLO dataset: ${error instanceof Error ? error.message : String(error)}`);
    throw error;
  }
};