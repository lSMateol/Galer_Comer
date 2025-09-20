// static/js/comparativo.js
window.__cmpInit = function () {
  console.log("comparativo.js (Chart.js): init");

  if (typeof Chart === "undefined") {
    console.error("comparativo.js: Chart.js no está cargado; se omiten gráficos");
    return;
  }

  const dataTag = document.getElementById("cmp-data");
  if (!dataTag) {
    console.warn("comparativo.js: no se encontró #cmp-data; nada que graficar");
    return;
  }

  let D = {};
  try { D = JSON.parse(dataTag.textContent || "{}"); }
  catch (e) { console.error("comparativo.js: JSON inválido en #cmp-data", e); return; }

  const L = Array.isArray(D.labels) ? D.labels.map(String) : [];
  if (L.length === 0) { console.warn("comparativo.js: labels vacío; no se graficará"); return; }

  const toNum = v => { const n = Number(v); return Number.isFinite(n) ? n : 0; };
  function normSeries(key) {
    const arr = Array.isArray(D[key]) ? D[key].map(toNum) : [];
    if (arr.length === L.length) return arr;
    if (arr.length > L.length) return arr.slice(0, L.length);
    return arr.concat(Array(L.length - arr.length).fill(0));
  }

  // --- Series normalizadas
  const be_inv_total = normSeries("be_inv_total");
  const be_inv_loc   = normSeries("be_inv_loc");
  const be_inv_parq  = normSeries("be_inv_parq");
  const be_inv_zonas = normSeries("be_inv_zonas");

  const be_ing_total = normSeries("be_ing_total");
  const be_ing_arr   = normSeries("be_ing_arr");
  const be_ing_adm   = normSeries("be_ing_adm");
  const be_ing_parq  = normSeries("be_ing_parq");

  const be_egr_total = normSeries("be_egr_total");
  const be_egr_mant  = normSeries("be_egr_mant");
  const be_egr_serv  = normSeries("be_egr_serv");
  const be_egr_sal   = normSeries("be_egr_sal");
  const be_egr_ope   = normSeries("be_egr_ope");
  const be_egr_adm   = normSeries("be_egr_adm");
  const be_egr_leg   = normSeries("be_egr_leg");
  const be_egr_imp   = normSeries("be_egr_imp");

  const bs_acces   = normSeries("bs_acces");   // decimales
  const bs_emp_dir = normSeries("bs_emp_dir"); // valores altos
  const bs_emp_ind = normSeries("bs_emp_ind"); // valores altos
  const bs_calidad = normSeries("bs_calidad"); // decimales

  const ar_af  = normSeries("ar_af");
  const ar_cp  = normSeries("ar_cp");
  const ar_nal = normSeries("ar_nal");
  const ar_sc  = normSeries("ar_sc");

  const idx_mun = normSeries("idx_mun");
  const idx_bc  = normSeries("idx_bc");
  const idx_bs_idx = normSeries("idx_bs_idx");
  const idx_fitness = normSeries("idx_fitness");

  // Helpers Chart.js
  function ctx(id) {
    const el = document.getElementById(id);
    if (!el) { console.warn(`comparativo.js: no existe canvas #${id}`); return null; }
    if (!el.style.height) el.style.height = "100%";
    return el.getContext("2d");
  }

  function makeBar(canvasId, label, data, opts = {}) {
    const c = ctx(canvasId);
    if (!c) return;
    return new Chart(c, {
      type: "bar",
      data: { labels: L, datasets: [{ label, data, borderWidth: 1 }] },
      options: Object.assign({
        responsive: true, maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { position: "top" }, tooltip: { mode: "index", intersect: false } }
      }, opts)
    });
  }

  function makeMultiBar(canvasId, datasets, opts = {}) {
    const c = ctx(canvasId);
    if (!c) return;
    const ds = (datasets || []).map(d => ({
      label: d.label || "",
      data: Array.isArray(d.data) ? d.data.map(toNum).slice(0, L.length) : Array(L.length).fill(0),
      borderWidth: 1, ...(d.other || {})
    }));
    return new Chart(c, {
      type: "bar",
      data: { labels: L, datasets: ds },
      options: Object.assign({
        responsive: true, maintainAspectRatio: false,
        scales: { x: { stacked: !!opts.stacked }, y: { beginAtZero: true, stacked: !!opts.stacked } },
        plugins: { legend: { position: "top" }, tooltip: { mode: "index", intersect: false } }
      }, opts)
    });
  }

  function makeRadar(canvasId, datasets, opts = {}) {
    const c = ctx(canvasId);
    if (!c) return;
    return new Chart(c, {
      type: "radar",
      data: {
        labels: L,
        datasets: (datasets || []).map(d => ({
          label: d.label || "",
          data: (Array.isArray(d.data) ? d.data : Array(L.length).fill(0)).slice(0, L.length),
          borderWidth: 1, fill: true
        }))
      },
      options: Object.assign({
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: {
          r: {
            beginAtZero: true,
            suggestedMax: Math.max(
              1,
              ...((datasets?.[0]?.data || []).map(toNum)),
              ...((datasets?.[1]?.data || []).map(toNum))
            ),
            ticks: { showLabelBackdrop: false }
          }
        }
      }, opts)
    });
  }

  // === Gráficos ===
  makeBar("chartInvTotal", "Inversión total", be_inv_total);

  makeMultiBar("chartInvDesglose", [
    { label: "Locales", data: be_inv_loc },
    { label: "Parqueaderos", data: be_inv_parq },
    { label: "Zonas comunes", data: be_inv_zonas }
  ], { stacked: true });

  makeMultiBar("chartIngEgrTotal", [
    { label: "Ingresos", data: be_ing_total },
    { label: "Egresos",  data: be_egr_total }
  ]);

  makeMultiBar("chartIngDesglose", [
    { label: "Arrendamiento", data: be_ing_arr },
    { label: "Administración", data: be_ing_adm },
    { label: "Parqueadero", data: be_ing_parq }
  ], { stacked: true });

  makeMultiBar("chartEgrDesglose", [
    { label: "Mantenimiento", data: be_egr_mant },
    { label: "Servicios públicos", data: be_egr_serv },
    { label: "Salarios", data: be_egr_sal },
    { label: "Operativos", data: be_egr_ope },
    { label: "Administrativos", data: be_egr_adm },
    { label: "Legales", data: be_egr_leg },
    { label: "Impuestos/Licencias", data: be_egr_imp }
  ], { stacked: true });

  // Beneficio Social — NUEVOS
  if (document.getElementById("chartEmpleo")) {
    makeMultiBar("chartEmpleo", [
      { label: "Empleo directo",   data: bs_emp_dir },
      { label: "Empleo indirecto", data: bs_emp_ind }
    ]);
  }
  if (document.getElementById("chartAccCal")) {
    makeRadar("chartAccCal", [
      { label: "Accesibilidad",   data: bs_acces },
      { label: "Calidad de vida", data: bs_calidad }
    ]);
  }
  // (Opcional) si mantienes un canvas con id="chartSocial", también lo dibuja:
  if (document.getElementById("chartSocial")) {
    makeMultiBar("chartSocial", [
      { label: "Accesibilidad",   data: bs_acces },
      { label: "Empleo directo",  data: bs_emp_dir },
      { label: "Empleo indirecto",data: bs_emp_ind },
      { label: "Calidad de vida", data: bs_calidad }
    ]);
  }

  if (document.getElementById("chartAreas")) {
    makeMultiBar(
      "chartAreas",
      [
        { label: "Alimentos frescos",    data: ar_af },
        { label: "Comidas preparadas",   data: ar_cp },
        { label: "No alimentarios",      data: ar_nal },
        { label: "Esp. complementarios", data: ar_sc }
      ],
      { stacked: true }
    );
  }

  makeRadar("chartIndices", [
    { label: "MUN", data: idx_mun },
    { label: "B/C", data: idx_bc },
    { label: "Benef. social", data: idx_bs_idx },
    { label: "Fitness", data: idx_fitness }
  ]);
};
