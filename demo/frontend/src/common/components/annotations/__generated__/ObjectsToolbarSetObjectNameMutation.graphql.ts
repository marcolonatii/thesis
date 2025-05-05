/**
 * @generated SignedSource<<1772cc0633540e47135da9df23bd3c3a>>
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
export type ObjectsToolbarSetObjectNameMutation$variables = {
  input: SetObjectNameInput;
};
export type ObjectsToolbarSetObjectNameMutation$data = {
  readonly setObjectName: {
    readonly name: string | null | undefined;
    readonly objectId: number;
    readonly success: boolean;
  };
};
export type ObjectsToolbarSetObjectNameMutation = {
  response: ObjectsToolbarSetObjectNameMutation$data;
  variables: ObjectsToolbarSetObjectNameMutation$variables;
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
    "name": "ObjectsToolbarSetObjectNameMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "ObjectsToolbarSetObjectNameMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "4010030c4495533e61869162d3f2e38a",
    "id": null,
    "metadata": {},
    "name": "ObjectsToolbarSetObjectNameMutation",
    "operationKind": "mutation",
    "text": "mutation ObjectsToolbarSetObjectNameMutation(\n  $input: SetObjectNameInput!\n) {\n  setObjectName(input: $input) {\n    success\n    objectId\n    name\n  }\n}\n"
  }
};
})();

(node as any).hash = "4d4c5921bd3e6e8af59dba719ea4282a";

export default node;
