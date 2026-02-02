export type ArtifactList = {
  files: string[];
};

export type AnalyzeRequest = {
  plots: boolean;

  seed?: number;
  nBoot?: number;
  alpha?: number;
  ciLevel?: number;

  // Step-1 contract params
  expectedSplit?: string; // e.g. "ad=0.96,psa=0.04"
  srmUniformTol?: number; // e.g. 0.02
  balanceSmdThreshold?: number; // e.g. 0.10
  balanceCramervThreshold?: number; // e.g. 0.10
  minUpliftAbs?: number; // e.g. 0.001
};

export type ABReport = Record<string, any>;
