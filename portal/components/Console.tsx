"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import {
  MODULES,
  hrefForModule,
  moduleFromPathname,
  type ModuleId,
} from "../lib/modules";
import type { ScienceBlock, ScienceView } from "../lib/science";

function Raster() {
  const rows = 4;
  const cols = 60;
  return (
    <div className="raster" aria-hidden="true">
      {Array.from({ length: rows }, (_, r) => (
        <div className="raster-row" key={r}>
          {Array.from({ length: cols }, (_, c) => (
            <i className="tick" key={c} />
          ))}
        </div>
      ))}
    </div>
  );
}

function QuestionList({ block }: { block: ScienceBlock }) {
  if (block.questions.length === 0) {
    return null;
  }
  return (
    <ol className="q-list">
      {block.questions.map((question) => (
        <li key={question}>{question}</li>
      ))}
    </ol>
  );
}

function CaptionRow({ block }: { block: ScienceBlock }) {
  if (block.captions.length === 0) {
    return null;
  }
  return (
    <ul className="captions">
      {block.captions.map((caption) => (
        <li key={caption}>{caption}</li>
      ))}
    </ul>
  );
}

export function Console({ science }: { science: ScienceView }) {
  const pathname = usePathname();
  const module: ModuleId = moduleFromPathname(pathname);

  return (
    <div className="console" data-layout="three-pane" data-module={module}>
      <header className="identity">
        <div className="brand">CALIBRATION TRAPS</div>
        <div className="role">PREFLIGHT INSTRUMENT</div>
        <div className="hold">HOLD / NOT LIVE INFERENCE</div>
      </header>

      <main className="panes" id="module-nomogram" hidden={module !== "nomogram"}>
        <section className="pane pane-controls" aria-labelledby="controls-title">
          <h2 id="controls-title">{science.nomogram.title}</h2>
          <p className="procedure">{science.nomogram.lead}</p>
          <dl className="params">
            <div>
              <dt>M</dt>
              <dd>HOLD</dd>
            </div>
            <div>
              <dt>q</dt>
              <dd>HOLD</dd>
            </div>
            <div>
              <dt>&alpha;</dt>
              <dd>HOLD</dd>
            </div>
            <div>
              <dt>K</dt>
              <dd>HOLD</dd>
            </div>
          </dl>
          <p className="formula">K &ge; (ln M &minus; ln &alpha;) / ln(1/q)</p>
          <CaptionRow block={science.nomogram} />
          <QuestionList block={science.nomogram} />
        </section>

        <section className="pane pane-scan" aria-labelledby="scan-title">
          <h2 id="scan-title">{science.scan.title}</h2>
          <p className="scan-caption">Checkpoint raster (structure only)</p>
          <Raster />
          <p className="procedure">{science.scan.lead}</p>
          <CaptionRow block={science.scan} />
          <QuestionList block={science.scan} />
        </section>

        <section className="pane pane-adjudication" aria-labelledby="bb-title">
          <h2 id="bb-title">{science.adjudication.title}</h2>
          <div className="bb-hero">
            <div className="confirm">
              <strong>HOLD</strong>
              <span>CONFIRM CHANNEL</span>
            </div>
            <div className="unpowered">
              <strong>HOLD</strong>
              <span>UNPOWERED CHANNEL</span>
            </div>
          </div>
          <p className="procedure">{science.adjudication.lead}</p>
          <CaptionRow block={science.adjudication} />
          <QuestionList block={science.adjudication} />
        </section>
      </main>

      <section className="module" id="module-testbed" hidden={module !== "testbed"}>
        <h2>{science.testbed.title}</h2>
        <p className="procedure">{science.testbed.lead}</p>
        <CaptionRow block={science.testbed} />
        <QuestionList block={science.testbed} />
      </section>

      <section className="module" id="module-drift" hidden={module !== "drift"}>
        <h2>{science.drift.title}</h2>
        <p className="procedure">{science.drift.lead}</p>
        <CaptionRow block={science.drift} />
        <QuestionList block={science.drift} />
      </section>

      <section className="module" id="module-bigbench" hidden={module !== "bigbench"}>
        <h2>{science.bigbench.title}</h2>
        <p className="procedure">{science.bigbench.lead}</p>
        <div className="bb-hero">
          <div className="confirm">
            <strong>HOLD</strong>
            <span>CONFIRM CHANNEL</span>
          </div>
          <div className="unpowered">
            <strong>HOLD</strong>
            <span>UNPOWERED CHANNEL</span>
          </div>
        </div>
        <CaptionRow block={science.bigbench} />
        <QuestionList block={science.bigbench} />
      </section>

      <section className="module" id="module-preflight" hidden={module !== "preflight"}>
        <h2>{science.preflight.title}</h2>
        <p className="procedure">{science.preflight.lead}</p>
        <dl className="params">
          <div>
            <dt>M</dt>
            <dd>scan length</dd>
          </div>
          <div>
            <dt>q</dt>
            <dd>exceedance rate</dd>
          </div>
          <div>
            <dt>&alpha;</dt>
            <dd>false-positive budget</dd>
          </div>
          <div>
            <dt>K</dt>
            <dd>run length</dd>
          </div>
        </dl>
        <CaptionRow block={science.preflight} />
        <QuestionList block={science.preflight} />
      </section>

      <section className="module" id="module-reproduce" hidden={module !== "reproduce"}>
        <h2>{science.rebuild.title}</h2>
        <p className="procedure">
          Clone{" "}
          <a href="https://github.com/PeterPonyu/calibration-traps">
            github.com/PeterPonyu/calibration-traps
          </a>
          . Rebuild from the experiment runners. Archival binaries:{" "}
          <a href="https://doi.org/10.5281/zenodo.21020386">
            10.5281/zenodo.21020386
          </a>
          .
        </p>
        <p className="note">{science.rebuild.lead}</p>
        <QuestionList block={science.rebuild} />
      </section>

      <nav className="fkey-rail" aria-label="modules">
        {MODULES.map((item) => {
          const href = hrefForModule(item.id);
          const active = item.id === module;
          return (
            <Link
              key={item.id}
              href={href}
              className={active ? "fkey is-active" : "fkey"}
              data-module={item.id}
              aria-current={active ? "page" : undefined}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      <footer className="site-foot">
        MIT · CC BY 4.0 ·{" "}
        <a href="https://doi.org/10.5281/zenodo.21020386">
          10.5281/zenodo.21020386
        </a>{" "}
        ·{" "}
        <a href="https://github.com/PeterPonyu/calibration-traps">
          github.com/PeterPonyu/calibration-traps
        </a>{" "}
        · sealed operating point · not a live detector
      </footer>
    </div>
  );
}
