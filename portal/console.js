/* Consumes papers/FIGURE-INDEX.json via the build copy at data/figures.json. */
fetch("data/figures.json")
  .then(function (response) {
    return response.ok ? response.json() : null;
  })
  .then(function (index) {
    if (!index || !Array.isArray(index.figures)) {
      return;
    }
    var scan = document.querySelector(".console-scan");
    if (!scan) {
      return;
    }
    var note = document.createElement("p");
    note.textContent = "INDEX figures: " + index.figures.length;
    scan.appendChild(note);
  })
  .catch(function () {
    /* Static console remains readable if data/ is absent in source preview. */
  });
