document.addEventListener("DOMContentLoaded", () => {
  const btn = document.querySelector("#btn-procesar");
  if (!btn) return;

  btn.addEventListener("click", (e) => {
    e.preventDefault();

    // Si tienes un <form id="param-form"> úsalo; si no, mandamos vacío.
    const form = document.querySelector("#param-form");
    const formData = form ? new FormData(form) : new FormData();

    // (Opcional) Si usas Flask-WTF/CSRF, toma el token:
    const csrfToken =
      document.querySelector("meta[name=csrf-token]")?.content ||
      document.querySelector("input[name=csrf_token]")?.value ||
      null;

    // Aviso visual (si ya tienes funciones/elementos de UI)
    if (typeof abrirModalProgreso === "function") abrirModalProgreso();

    fetch("/procesar_todas_galerias", {
      method: "POST",
      body: formData,
      credentials: "same-origin", // asegura cookies/sesión
      headers: csrfToken ? { "X-CSRFToken": csrfToken } : undefined,
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status !== "ok") {
          alert(data.message || "Error inesperado");
          return;
        }

        if (data.completed) {
          // MODO SÍNCRONO (en Vercel con SYNC_MODE=1): ya terminó.
          window.location.href = `/resultados?run_id=${encodeURIComponent(
            data.run_id
          )}`;
        } else {
          // MODO ASÍNCRONO (tu flujo actual con threads y polling)
          if (typeof empezarPollingLogs === "function") {
            empezarPollingLogs(data.thread_id);
          } else {
            // Fallback por si no tienes esa función
            console.log("thread_id:", data.thread_id);
          }
        }
      })
      .catch((err) => {
        console.error(err);
        alert("Error de red o servidor");
      });
  });
});
