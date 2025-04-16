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
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import useVideo from '@/common/components/video/editor/useVideo';
import {activeTrackletObjectIdAtom, labelTypeAtom, sessionAtom, trackletNamesAtom} from '@/demo/atoms'; // Added sessionAtom
import {Add} from '@carbon/icons-react';
import {useAtom, useAtomValue, useSetAtom} from 'jotai'; // Added useSetAtom, useAtomValue
import {useState} from 'react';
import ObjectNameModal from '@/common/components/annotations/ObjectNameModal';
import { graphql, useMutation } from 'react-relay'; // Added Relay imports
import type { SetObjectNameMutation } from '@/graphql/mutations/__generated__/SetObjectNameMutation.graphql'; // Added type import

export default function AddObjectButton() {
  const video = useVideo();
  const [trackletNames, setTrackletNames] = useAtom(trackletNamesAtom);
  const [activeTrackletId, setActiveTrackletId] = useAtom(activeTrackletObjectIdAtom);
  const setLabelType = useSetAtom(labelTypeAtom); // Changed from useAtom[1] to useSetAtom
  const session = useAtomValue(sessionAtom); // Get session state
  const {enqueueMessage, enqueueError} = useMessagesSnackbar(); // Added enqueueError
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [objectName, setObjectName] = useState('');

  // Setup GraphQL Mutation
  const [commitSetObjectName, isSettingName] = useMutation<SetObjectNameMutation>(graphql`
    mutation AddObjectButtonSetObjectNameMutation($input: SetObjectNameInput!) {
      setObjectName(input: $input) {
        success
        objectId
        name
      }
    }
  `);

  function handleOpenModal() {
    setObjectName(''); // Reset name field when opening for a new object
    setIsModalOpen(true);
  }

  async function handleConfirmName(name: string) {
    setIsModalOpen(false); // Close modal first
    enqueueMessage('addObjectClick');

    // 1. Create the tracklet locally
    const tracklet = await video?.createTracklet();
    if (tracklet == null) {
        enqueueError('Failed to create new object locally.');
        setObjectName(''); // Reset name state
        return;
    }

    setActiveTrackletId(tracklet.id);
    setLabelType('positive');
    const finalName = name.trim();

    // 2. Update local Jotai state immediately
    if (finalName) {
      // Use functional update for atom
      setTrackletNames((prev) => ({ ...prev, [tracklet.id]: finalName }));
    }
    // If name is empty, default name "Object X" will be used by getObjectLabel

    // 3. If a name was provided, call the backend mutation
    if (finalName) {
      if (!session?.id) {
        console.error('Cannot set name for new object, session ID is missing');
        enqueueError('Failed to save name for new object: Session not found.');
        // Keep local changes, but backend won't be updated
      } else {
          commitSetObjectName({
          variables: {
            input: {
              sessionId: session.id,
              objectId: tracklet.id,
              name: finalName,
            },
          },
          onCompleted: (response, errors) => {
            if (errors) {
              console.error('Error setting name for new object:', errors);
              enqueueError(`Failed to save name for new object: ${errors[0].message}`);
              // Optionally revert local state
            } else if (response.setObjectName?.success) {
              console.log(`Successfully set name for new object ${response.setObjectName.objectId} to '${response.setObjectName.name}'`);
              // Ensure local state matches backend (should already match if finalName was set)
              const backendName = response.setObjectName.name;
               setTrackletNames((prev) => {
                    const updatedNames = { ...prev };
                    if (backendName) {
                        updatedNames[tracklet.id] = backendName;
                    } else {
                        // Should not happen if finalName was non-empty, but handle defensively
                        delete updatedNames[tracklet.id];
                    }
                    return updatedNames;
                });
            } else {
              console.error('Failed to set name for new object (backend reported failure):', response);
              enqueueError('Failed to save name for new object on the server.');
              // Optionally revert local state
            }
          },
          onError: (error) => {
            console.error('Network/GraphQL error setting name for new object:', error);
            enqueueError(`Failed to save name for new object: ${error.message}`);
            // Optionally revert local state
          },
        });
      }
    }

    // 4. Reset object name state after processing
    setObjectName('');
  }

  function handleCancel() {
    setIsModalOpen(false);
    setObjectName('');
  }

  return (
    <>
      <div
        onClick={handleOpenModal}
        className="group flex justify-start mx-4 px-4 py-4 bg-transparent text-white !rounded-xl border-none cursor-pointer hover:bg-graydark-800/50 transition-colors duration-150" // Added padding and hover effect
        role="button" // Semantics
        tabIndex={0} // Make it focusable
        onKeyDown={(e) => e.key === 'Enter' && handleOpenModal()} // Keyboard accessibility
      >
        <div className="flex gap-6 items-center">
          <div className=" group-hover:bg-graydark-700 border border-white relative h-12 w-12 md:w-20 md:h-20 shrink-0 rounded-lg flex items-center justify-center transition-colors duration-150">
            <Add size={36} className="group-hover:text-white text-gray-300 transition-colors duration-150" />
          </div>
          <div className="font-medium text-base">Add another object</div>
        </div>
      </div>
      <ObjectNameModal
        isOpen={isModalOpen}
        onConfirm={handleConfirmName}
        onCancel={handleCancel}
        value={objectName}
        onChange={setObjectName}
        // Explicitly set props for adding a new object
        modalTitle="Name your new object"
        confirmButtonText="Create Object"
      />
    </>
  );
}