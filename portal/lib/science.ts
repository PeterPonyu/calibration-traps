export type ScienceBlock = {
  title: string;
  lead: string;
  captions: string[];
  questions: string[];
};

export type ScienceView = {
  nomogram: ScienceBlock;
  scan: ScienceBlock;
  adjudication: ScienceBlock;
  testbed: ScienceBlock;
  drift: ScienceBlock;
  bigbench: ScienceBlock;
  preflight: ScienceBlock;
  rebuild: ScienceBlock;
};

export const EMPTY_SCIENCE: ScienceView = {
  nomogram: {
    title: "NOMOGRAM CONTROLS",
    lead: "Closed-form nomogram selects K from M, q, and false-positive budget α. Not a live detector.",
    captions: [],
    questions: [],
  },
  scan: {
    title: "SCAN / RUN-LENGTH",
    lead: "Checkpoint raster and sustained-K ladder. Counts stay off this door.",
    captions: [],
    questions: [],
  },
  adjudication: {
    title: "ADJUDICATION",
    lead: "Confirm and unpowered channels. Task rows stay off this door.",
    captions: [],
    questions: [],
  },
  testbed: {
    title: "TESTBED",
    lead: "Supervised-fit control: can the same transformer fit labels without a timing claim?",
    captions: [],
    questions: [],
  },
  drift: {
    title: "DRIFT",
    lead: "Scan grid and budget grid. Heatmap artwork is a rebuild product, not a live run.",
    captions: [],
    questions: [],
  },
  bigbench: {
    title: "BIG-BENCH",
    lead: "Look-elsewhere adjudication. Channel structure only; outcome counts stay off this door.",
    captions: [],
    questions: [],
  },
  preflight: {
    title: "PREFLIGHT",
    lead: "Pick K from M, q, and α before reading a timing. This page does not run the detector.",
    captions: [],
    questions: [],
  },
  rebuild: {
    title: "REPRODUCE-AS-REBUILD",
    lead: "Clone the repo, run the seeded experiment scripts, and compare against committed logs.",
    captions: [],
    questions: [
      "Clone github.com/PeterPonyu/calibration-traps",
      "Rebuild from the experiment runners",
      "Compare against committed per-run logs",
      "Archival binaries: 10.5281/zenodo.21020386",
    ],
  },
};
