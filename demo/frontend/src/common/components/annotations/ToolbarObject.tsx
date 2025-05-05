/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import ObjectActions from '@/common/components/annotations/ObjectActions';
import ObjectPlaceholder from '@/common/components/annotations/ObjectPlaceholder';
import ObjectThumbnail from '@/common/components/annotations/ObjectThumbnail';
import ToolbarObjectContainer from '@/common/components/annotations/ToolbarObjectContainer';
import useVideo from '@/common/components/video/editor/useVideo';
// Import the exported type and hook
import { ExtendedTracklet, useObjectLabel } from '@/common/components/annotations/ObjectUtils';
import emptyFunction from '@/common/utils/emptyFunction';
import {activeTrackletObjectIdAtom, trackletNamesAtom} from '@/demo/atoms';
import {useAtom, useSetAtom} from 'jotai';
import useReportError from '@/common/error/useReportError';
import { Edit } from '@carbon/icons-react'; // Import Edit icon
import { Button } from 'react-daisyui'; // Import Button for the edit action

// Remove local type definition, use imported one
// type ExtendedTracklet = BaseTracklet & { name?: string | null };

type Props = {
  // label prop is no longer needed, we'll derive it using the hook
  tracklet: ExtendedTracklet;
  isActive: boolean;
  isMobile?: boolean;
  onClick?: () => void;
  onThumbnailClick?: () => void;
  // Add callback prop for initiating edit
  onEditName: (trackletId: number, currentName: string) => void;
};

export default function ToolbarObject({
  // Remove label from props destructuration
  tracklet,
  isActive,
  isMobile = false,
  onClick,
  onThumbnailClick = emptyFunction,
  onEditName, // Add new prop
}: Props) {
  const video = useVideo();
  const setActiveTrackletId = useSetAtom(activeTrackletObjectIdAtom);
  const [trackletNames, setTrackletNames] = useAtom(trackletNamesAtom); // Get names state
  const reportError = useReportError();
  const label = useObjectLabel(tracklet); // Use the hook to get the label

  async function handleCancelNewObject() {
    try {
      // Ensure tracklet and id are valid before deleting
      if (tracklet?.id != null) {
        await video?.deleteTracklet(tracklet.id);
        setTrackletNames((prev) => {
          const newNames = { ...prev };
          delete newNames[tracklet.id];
          return newNames;
        });
      }
    } catch (error) {
      reportError(error);
    } finally {
      setActiveTrackletId(null);
    }
  }

  // Handler for the edit button click
  function handleEditClick(event: React.MouseEvent) {
      event.stopPropagation(); // Prevent the container's onClick from firing
      if (tracklet?.id != null) {
          onEditName(tracklet.id, label); // Pass ID and current label
      }
  }

  if (!tracklet.isInitialized) {
    return (
      <ToolbarObjectContainer
        alignItems="center"
        isActive={isActive}
        title="New object"
        subtitle="Click an object in the video to start." // Updated subtitle
        thumbnail={<ObjectPlaceholder showPlus={false} />}
        isMobile={isMobile}
        // Don't trigger main onClick when cancelling
        onClick={onClick}
        onCancel={(e) => { e?.stopPropagation(); handleCancelNewObject(); }}
      />
    );
  }

  return (
    <ToolbarObjectContainer
      isActive={isActive}
      onClick={onClick}
      // Title section now includes the edit button
      title={
        <div className="flex items-center gap-2">
          <span>{label}</span>
           {/* Show edit button only when active and not mobile (or adjust as needed) */}
          {isActive && !isMobile && (
            <Button
              size="sm"
              shape="circle"
              color="ghost"
              className="!p-1 !min-h-0 !h-6 !w-6 text-gray-400 hover:text-white hover:bg-graydark-700"
              onClick={handleEditClick}
              aria-label={`Edit name for ${label}`} // Accessibility
              title={`Edit name for ${label}`} // Tooltip
            >
              <Edit size={16} />
            </Button>
          )}
        </div>
      }
      subtitle="" // Subtitle not needed when initialized
      thumbnail={
        <ObjectThumbnail
          thumbnail={tracklet.thumbnail}
          color={tracklet.color}
          onClick={onThumbnailClick}
        />
      }
      isMobile={isMobile}>
      <ObjectActions objectId={tracklet.id} active={isActive} />
    </ToolbarObjectContainer>
  );
}