/**
 * @generated SignedSource<<a441099cf81cc0d66f9f525e2228d5e8>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type SetObjectNameInput = {
  name: string;
  objectId: number;
  sessionId: string;
};
export type AddObjectButtonSetObjectNameMutation$variables = {
  input: SetObjectNameInput;
};
export type AddObjectButtonSetObjectNameMutation$data = {
  readonly setObjectName: {
    readonly name: string | null | undefined;
    readonly objectId: number;
    readonly success: boolean;
  };
};
export type AddObjectButtonSetObjectNameMutation = {
  response: AddObjectButtonSetObjectNameMutation$data;
  variables: AddObjectButtonSetObjectNameMutation$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "defaultValue": null,
    "kind": "LocalArgument",
    "name": "input"
  }
],
v1 = [
  {
    "alias": null,
    "args": [
      {
        "kind": "Variable",
        "name": "input",
        "variableName": "input"
      }
    ],
    "concreteType": "SetObjectNameResponse",
    "kind": "LinkedField",
    "name": "setObjectName",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "success",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "objectId",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "name",
        "storageKey": null
      }
    ],
    "storageKey": null
  }
];
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "AddObjectButtonSetObjectNameMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "AddObjectButtonSetObjectNameMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "f7325e8f53f9b7c76d5c59bccc435631",
    "id": null,
    "metadata": {},
    "name": "AddObjectButtonSetObjectNameMutation",
    "operationKind": "mutation",
    "text": "mutation AddObjectButtonSetObjectNameMutation(\n  $input: SetObjectNameInput!\n) {\n  setObjectName(input: $input) {\n    success\n    objectId\n    name\n  }\n}\n"
  }
};
})();

(node as any).hash = "84f86cf7820511efbc8faa4c00a4f53e";

export default node;
