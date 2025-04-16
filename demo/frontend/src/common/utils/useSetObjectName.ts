// File: /home/david_elliott/github/sam2-git/demo/frontend/src/common/utils/useSetObjectName.ts
import { useCallback } from 'react';
import { useMutation } from 'react-relay';
import Logger from '@/common/logger/Logger';

// Import the mutation defined with the graphql tag
import { SET_OBJECT_NAME_MUTATION } from '@/common/api/mutations';

// Import generated types for type safety (Make sure Relay compiler has run)
// The specific import name depends on your Relay configuration and file structure.
// It often follows the pattern: FileNameMutationNameMutation.graphql
import { mutationsSetObjectNameMutation } from './__generated__/mutationsSetObjectNameMutation.graphql';

// Define the input type for the hook function for clarity
type SetObjectNameInput = {
  sessionId: string;
  objectId: number;
  name: string;
};

// Define the return type of the hook
type UseSetObjectNameReturn = {
  setObjectName: (input: SetObjectNameInput) => Promise<boolean>; // Return promise indicating success/failure
  loading: boolean;
};

/**
 * Custom hook to handle the setObjectName mutation via Relay.
 *
 * @returns An object containing:
 * - setObjectName: An async function to call the mutation.
 * - loading: A boolean indicating if the mutation is in flight.
 */
export function useSetObjectName(): UseSetObjectNameReturn {
  // Use the Relay mutation hook
  const [commitMutation, isMutationInFlight] = useMutation<mutationsSetObjectNameMutation>(SET_OBJECT_NAME_MUTATION);

  const setObjectName = useCallback(
    async (input: SetObjectNameInput): Promise<boolean> => {
      Logger.info('Calling setObjectName Relay mutation with input:', input);

      return new Promise((resolve) => {
        commitMutation({
          variables: {
            input: { // The input object structure must match the $input variable in the mutation
              sessionId: input.sessionId,
              objectId: input.objectId,
              name: input.name,
            },
          },
          onCompleted: (response, errors) => {
            // Handle Relay errors array
            if (errors && errors.length > 0) {
              Logger.error('Error(s) completing setObjectName Relay mutation:', errors);
              resolve(false); // Indicate failure
              return;
            }

            // Check the response payload structure defined in the mutation
            const success = response?.setObjectName?.success ?? false;
            if (success) {
              Logger.info('Successfully set object name on server via Relay.');
            } else {
              // This case might occur if the server mutation logic returns success: false
              // or if the response structure doesn't match expectations.
              Logger.warn('Server responded with success=false or unexpected payload for setObjectName.', response);
            }
            resolve(success); // Resolve with the success status from the backend
          },
          onError: (error) => {
            // Handle network/graphql execution errors
            Logger.error('Network or server error during setObjectName Relay mutation:', error);
            resolve(false); // Indicate failure
          },
          // Optional: Optimistic response or updater functions can be added here if needed
          // optimisticResponse: {
          //   setObjectName: {
          //      success: true // Assume success optimistically
          //   }
          // },
          // updater: (store) => {
          //  // Logic to update the Relay store manually if needed
          // },
        });
      });
    },
    [commitMutation] // Dependency array includes the commit function
  );

  return {
    setObjectName,
    loading: isMutationInFlight,
  };
}