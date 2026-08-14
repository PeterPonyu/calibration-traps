export const MODULES = [
  { id: "nomogram", label: "Nomogram" },
  { id: "testbed", label: "Testbed" },
  { id: "drift", label: "Drift" },
  { id: "bigbench", label: "BIG-Bench" },
  { id: "preflight", label: "Preflight" },
  { id: "reproduce", label: "Reproduce" },
] as const;

export type ModuleId = (typeof MODULES)[number]["id"];

export const MODULE_IDS: readonly ModuleId[] = MODULES.map((item) => item.id);

export function isModuleId(value: string): value is ModuleId {
  return (MODULE_IDS as readonly string[]).includes(value);
}

/** Map a Next.js pathname (basePath already stripped) to a console module. */
export function moduleFromPathname(pathname: string | null | undefined): ModuleId {
  const parts = (pathname ?? "/").split("/").filter(Boolean);
  const start = parts[0] === "calibration-traps" ? 1 : 0;
  const slug = parts[start];
  if (slug === "big-bench") {
    return "bigbench";
  }
  if (slug && isModuleId(slug)) {
    return slug;
  }
  return "nomogram";
}

export function hrefForModule(id: ModuleId): string {
  return `/${id}/`;
}
