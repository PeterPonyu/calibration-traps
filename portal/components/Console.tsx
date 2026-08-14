"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import {
  MODULES,
  hrefForModule,
  moduleFromPathname,
  type ModuleId,
} from "../lib/modules";

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

export function Console() {
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
          <h2 id="controls-title">DOCUMENTED CONTROLS</h2>
          <p className="procedure">
            Closed-form nomogram selects K from M, q, and false-positive budget
            &alpha;. Static warehouse JSON. Not a live detector.
          </p>
          <dl className="params">
            <div>
              <dt>M</dt>
              <dd>SEALED</dd>
            </div>
            <div>
              <dt>q</dt>
              <dd>SEALED</dd>
            </div>
            <div>
              <dt>&alpha;</dt>
              <dd>SEALED</dd>
            </div>
            <div>
              <dt>K</dt>
              <dd>SEALED</dd>
            </div>
          </dl>
          <p className="formula">K &ge; (ln M &minus; ln &alpha;) / ln(1/q)</p>
          <p className="sealed">Operating point lives in warehouse JSON. HOLD.</p>
        </section>

        <section className="pane pane-scan" aria-labelledby="scan-title">
          <h2 id="scan-title">SCAN / RUN-LENGTH</h2>
          <p className="scan-caption">Checkpoint raster (structure only)</p>
          <Raster />
          <h3 className="ladder-title">K-ladder (exceedance runs &ge; K)</h3>
          <ol className="ladder">
            <li>
              <span className="k">K=1</span>
              <span className="count">SEALED</span>
            </li>
            <li>
              <span className="k">K=2</span>
              <span className="count">SEALED</span>
            </li>
            <li>
              <span className="k">K=3</span>
              <span className="count">SEALED</span>
            </li>
            <li>
              <span className="k">K=5</span>
              <span className="count">SEALED</span>
            </li>
          </ol>
          <p className="note">
            Warehouse summaries, not pasted PDFs. Scan counts are not on this
            door.
          </p>
        </section>

        <section className="pane pane-adjudication" aria-labelledby="bb-title">
          <h2 id="bb-title">ADJUDICATION</h2>
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
          <table className="bb-table">
            <thead>
              <tr>
                <th>TASK</th>
                <th>JUMP</th>
                <th>
                  E<sub>task</sub>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={3}>SEALED — rows live in warehouse INDEX</td>
              </tr>
            </tbody>
          </table>
          <p className="note">Look-elsewhere table is not published on this door.</p>
        </section>
      </main>

      <section className="module" id="module-testbed" hidden={module !== "testbed"}>
        <h2>TESTBED</h2>
        <p className="procedure">
          Supervised-fit control is a warehouse JSON pointer, not a live run.
        </p>
        <p className="note">
          Summary: <span className="mono">figs/summaries/E2_supervised_fit.json</span>
        </p>
      </section>

      <section className="module" id="module-drift" hidden={module !== "drift"}>
        <h2>DRIFT</h2>
        <p className="procedure">
          Scan / budget grids are warehouse JSON. Heatmap artwork is an untracked
          vec-tier include, not a summary file.
        </p>
        <p className="note">
          <span className="mono">E2_td_grid.json</span> ·{" "}
          <span className="mono">E2_budget_grid.json</span>
        </p>
      </section>

      <section className="module" id="module-bigbench" hidden={module !== "bigbench"}>
        <h2>BIG-BENCH</h2>
        <p className="procedure">
          Adjudication structure only. Task rows and channel counts stay in the
          warehouse, not on this door.
        </p>
      </section>

      <section className="module" id="module-preflight" hidden={module !== "preflight"}>
        <h2>PREFLIGHT</h2>
        <p className="procedure">
          Documented instrument: pick K from M, q, and &alpha; before reading a
          timing. This page does not run the detector.
        </p>
      </section>

      <section className="module" id="module-reproduce" hidden={module !== "reproduce"}>
        <h2>REPRODUCE</h2>
        <p className="procedure">
          Clone{" "}
          <a href="https://github.com/PeterPonyu/calibration-traps">
            github.com/PeterPonyu/calibration-traps
          </a>
          . Rebuild figures via <span className="mono">papers/figs/PIPELINE.md</span>.
          Archival binaries:{" "}
          <a href="https://doi.org/10.5281/zenodo.21020386">
            10.5281/zenodo.21020386
          </a>
          .
        </p>
        <p className="note">
          Portal contract: <span className="mono">papers/FIGURE-INDEX.json</span>{" "}
          (copied to <span className="mono">data/figures.json</span>). Pointer tex:{" "}
          <span className="mono">papers/E2/main.tex</span>.
        </p>
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
              <kbd>{item.key}</kbd> {item.label}
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
        · papers/FIGURE-INDEX.json · papers/figs/PIPELINE.md · papers/E2/main.tex
      </footer>
    </div>
  );
}
