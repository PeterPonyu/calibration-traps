import index from "../public/data/figures.json";
import nomogram from "../public/data/figs/summaries/E2_nomogram.json";
import tdGrid from "../public/data/figs/summaries/E2_td_grid.json";
import budget from "../public/data/figs/summaries/E2_budget_grid.json";
import fit from "../public/data/figs/summaries/E2_supervised_fit.json";
import induction from "../public/data/figs/summaries/E2_induction.json";
import caseStudy from "../public/data/figs/summaries/E2_case.json";
import specroute from "../public/data/figs/summaries/E2_specroute.json";

import { EMPTY_SCIENCE, type ScienceView } from "./science";

type Panel = { question?: string };
type Summary = { panels?: Panel[] };
type Figure = { id?: string; caption_panels?: string[] };
type IndexPayload = { figures?: Figure[] };

function cleanCaption(value: string): string {
  return value.replace(/\$/g, "").trim();
}

function questionsOf(summary: Summary): string[] {
  return (summary.panels ?? [])
    .map((panel) => (panel.question ?? "").trim())
    .filter(Boolean);
}

function captionsOf(payload: IndexPayload, id: string): string[] {
  const figure = (payload.figures ?? []).find((row) => row.id === id);
  return (figure?.caption_panels ?? []).map(cleanCaption).filter(Boolean);
}

export function loadScience(): ScienceView {
  const catalog = index as IndexPayload;
  return {
    nomogram: {
      ...EMPTY_SCIENCE.nomogram,
      captions: captionsOf(catalog, "E2_nomogram"),
      questions: questionsOf(nomogram),
    },
    scan: {
      ...EMPTY_SCIENCE.scan,
      captions: captionsOf(catalog, "E2_td_grid"),
      questions: questionsOf(tdGrid),
    },
    adjudication: {
      ...EMPTY_SCIENCE.adjudication,
      captions: captionsOf(catalog, "E2_case"),
      questions: questionsOf(caseStudy),
    },
    testbed: {
      ...EMPTY_SCIENCE.testbed,
      captions: captionsOf(catalog, "E2_supervised_fit"),
      questions: questionsOf(fit),
    },
    drift: {
      ...EMPTY_SCIENCE.drift,
      captions: [
        ...captionsOf(catalog, "E2_td_grid"),
        ...captionsOf(catalog, "E2_budget_grid"),
      ],
      questions: [...questionsOf(tdGrid), ...questionsOf(budget)],
    },
    bigbench: {
      ...EMPTY_SCIENCE.bigbench,
      captions: [
        ...captionsOf(catalog, "E2_induction"),
        ...captionsOf(catalog, "E2_case"),
      ],
      questions: [...questionsOf(induction), ...questionsOf(caseStudy)],
    },
    preflight: {
      ...EMPTY_SCIENCE.preflight,
      captions: [
        ...captionsOf(catalog, "E2_nomogram"),
        ...captionsOf(catalog, "E2_specroute"),
      ],
      questions: [...questionsOf(nomogram), ...questionsOf(specroute)],
    },
    rebuild: EMPTY_SCIENCE.rebuild,
  };
}
