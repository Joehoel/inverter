import { SSTConfig } from "sst";
import { API } from "./stacks/API";
import { Storage } from "./stacks/Storage";
import { Web } from "./stacks/Web";

export default {
  config(_input) {
    return {
      name: "inverter",
      region: "us-east-1",
    };
  },
  stacks(app) {
    app.stack(Storage).stack(API).stack(Web);
  },
} satisfies SSTConfig;
