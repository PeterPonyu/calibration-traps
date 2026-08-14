import { loadScience } from "../lib/loadScience";
import { Console } from "./Console";

export function ConsoleFrame() {
  const science = loadScience();
  return <Console science={science} />;
}
