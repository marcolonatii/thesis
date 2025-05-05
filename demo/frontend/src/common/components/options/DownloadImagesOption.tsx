/**
 * Option Button to download original images as a Zip archive.
 */
import { ImageSearch } from '@carbon/icons-react'; // Example icon
import { useState } from 'react';
import { useAtomValue } from 'jotai';
import { sessionAtom } from '@/demo/atoms';
import { downloadImagesZip } from '@/common/api/downloadApi';
import OptionButton from './OptionButton';
import Logger from '@/common/logger/Logger';

export default function DownloadImagesOption() {
  const session = useAtomValue(sessionAtom);
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async () => {
    // Prevent action if already loading
    if (isLoading) {
      Logger.warn('Download Images cancelled: Already loading.');
      return;
    }
    // Check for session ID *before* starting the download
    // This check is still useful for logging/early return, even with isDisabled updated
    if (!session?.session_id) {
      Logger.warn('Download Images cancelled: No session ID available yet.');
      // Optional: Add user feedback like an alert or snackbar here
      alert('Session data is not yet available. Please wait for processing to complete.');
      return;
    }

    setIsLoading(true);
    try {
      await downloadImagesZip(session.session_id);
    } catch (error) {
      // Error already logged and alerted in API function
      Logger.error('Download Images Option caught error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <OptionButton
      title="Download Images (Zip)"
      Icon={ImageSearch} // Choose an appropriate icon
      loadingProps={{
        loading: isLoading,
        label: 'Downloading...',
      }}
      onClick={handleClick}
    />
  );
}