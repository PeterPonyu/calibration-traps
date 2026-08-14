export const MODULES = [
  { id: "nomogram", key: "F1", label: "Nomogram" },
  { id: "testbed", key: "F2", label: "Testbed" },
  { id: "drift", key: "F3", label: "Drift" },
  { id: "bigbench", key: "F4", label: "BIG-Bench" },
  { id: "preflight", key: "F5", label: "Preflight" },
  { id: "reproduce", key: "F6", label: "Reproduce" },
] as const;

export type ModuleId = (typeof MODULES)[number]["id"];

export const MODULE_IDS: readonly ModuleId[] = MODULES.map((item) => item.id);

export function isModuleId(value: string): value is ModuleId {
  return (MODULE_IDS as readonly string[]).includes(value);
}

/** Map a Next.js pathname (basePath already stripped) to an F-key module. */
export function moduleFromPathname(pathname: string | null | undefined): ModuleId {
  const parts = (pathname ?? "/").split("/").filter(Boolean);
  const start = parts[0] === "calibration-traps" ? 1 : 0;
  const slug = parts[start];
  if (slug && isModuleId(slug)) {
    return slug;
  }
  return "nomogram";
}

export function hrefForModule(id: ModuleId): string {
  return `/${id}/`;
}
