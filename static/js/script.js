(() => {
  document.addEventListener('DOMContentLoaded', () => {
    // 1) Validación del formulario y UX de envío
    const form = document.getElementById('parametrosForm');
    if (form) {
      form.addEventListener('submit', (e) => {
        const pesoBS = parseInt(document.getElementById('pesoBS').value);
        const pesoBE = parseInt(document.getElementById('pesoBE').value);
        const pesoMUN = parseInt(document.getElementById('pesoMUN').value);

        if (pesoBS + pesoBE + pesoMUN !== 100) {
          e.preventDefault();
          alert('La suma de los pesos debe ser igual a 100%');
          return;
        }
        for (let i = 1; i <= 7; i++) {
          const tamLote = document.getElementById('tamLote' + i);
          const canPri  = document.getElementById('canPri' + i);
          const canSec  = document.getElementById('canSec' + i);
          if (!tamLote?.value || !canPri?.value || !canSec?.value) {
            e.preventDefault();
            alert('Por favor complete todos los campos de la Galería ' + (i <= 6 ? i : 'Nueva'));
            return;
          }
        }
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
          submitBtn.disabled = true;
          submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Iniciando procesamiento...';
        }
      });
    }

    // 2) Si hay modal de progreso, arrancamos el polling y el dock de logs
    const progressModal = document.getElementById('progressModal');
    const progressData  = document.getElementById('progressData');

    if (!progressModal || !progressData) return;

    // Mostrar panel de logs
    const logsDock     = document.getElementById('logsDock');
    const logPre       = document.getElementById('logPre');
    const statusBadge  = document.getElementById('execStatus');
    const clearBtn     = document.getElementById('clearLogs');
    if (logsDock) logsDock.style.display = 'block';

    let threadId = progressData.getAttribute('data-thread-id') || '';
    const runId  = progressData.getAttribute('data-run-id') || ''; 
    if (!threadId) {
      // fallback a querystring
      const urlParams = new URLSearchParams(window.location.search);
      threadId = urlParams.get('thread_id') || '';
    }
    const resultsUrl = progressData.getAttribute('data-results-url') || '/resultados';
    const totalGalleries = parseInt(progressData.getAttribute('data-total-galleries') || '7', 10);

    // Barras del modal
    const overallBar = document.getElementById('overallProgress');
    const finalMsg   = document.getElementById('finalMessage');
    const bestComunaInfo = document.getElementById('bestComunaInfo');
    const bestComunaText = document.getElementById('bestComunaText');

    let lastLen = 0;        // cuántas líneas de logs ya mostré
    let completed = 0;      // cuántas galerías con ROI ya registradas
    const roiSeen = {};     // para contabilizar una sola vez por galería

    clearBtn?.addEventListener('click', () => {
      if (logPre) logPre.textContent = '';
      lastLen = 0;
    });

    // Utilidades
    function parseRoiFromLogs(logs) {
      const out = {};
      const re = /^GALERIA_(\d+)_ROI:(-?\d+(\.\d+)?)/i;
      logs.forEach(line => {
        const m = line.match(re);
        if (m) out[parseInt(m[1],10)] = parseFloat(m[2]);
      });
      return out;
    }
    function parseBestComuna(logs) {
      const re = /Mejor comuna encontrada:\s*(\d+)/i;
      for (let i = logs.length - 1; i >= 0; i--) {
        const m = logs[i].match(re);
        if (m) return parseInt(m[1],10);
      }
      return null;
    }
    function updateGalleryBars(roiByGallery) {
      for (let i = 1; i <= totalGalleries; i++) {
        const bar   = document.getElementById(`progress-${i}`);
        const label = document.getElementById(`progress-text-${i}`);
        const roiEl = document.getElementById(`roi-${i}`);
        if (!bar || !label) continue;

        if (roiByGallery[i] !== undefined) {
          if (!roiSeen[i]) {
            roiSeen[i] = true;
            completed++;
          }
          bar.style.width = '100%';
          bar.classList.add('bg-success');
          bar.classList.remove('progress-bar-animated');
          label.textContent = 'Completado';
          if (roiEl) roiEl.textContent = `${roiByGallery[i].toFixed(2)}%`;
        } else {
          // estado intermedio si aún no hay ROI para esa galería
          if (label.textContent !== 'Completado') {
            bar.style.width = '40%';
            bar.classList.add('progress-bar-animated');
            label.textContent = 'Procesando…';
          }
        }
      }
      // barra general
      const pct = Math.min(100, Math.round((completed / totalGalleries) * 100));
      if (overallBar) {
        overallBar.style.width = `${pct}%`;
        overallBar.textContent = `${pct}%`;
      }
    }

    async function poll() {
      try {
        const res = await fetch(`/api/logs/${threadId}`);
        const data = await res.json();
        const logs = data.logs || [];
        const status = data.status || 'ejecutando';

        // Estado en badge
        if (statusBadge) {
          statusBadge.textContent = status;
          statusBadge.classList.toggle('bg-success', status === 'completado');
          statusBadge.classList.toggle('bg-secondary', status !== 'completado');
        }

        // Agregar logs nuevos al dock
        if (logPre && logs.length > lastLen) {
          const newLines = logs.slice(lastLen).join('\n');
          logPre.textContent += (logPre.textContent ? '\n' : '') + newLines;
          logPre.scrollTop = logPre.scrollHeight;
          lastLen = logs.length;
        }

        // Actualizar barras por ROI detectado
        const roiByGallery = parseRoiFromLogs(logs);
        updateGalleryBars(roiByGallery);

        // Mostrar comuna óptima si está en logs
        const best = parseBestComuna(logs);
        if (best && bestComunaInfo && bestComunaText) {
          bestComunaText.textContent = `Mejor comuna: ${best} — la nueva galería se optimizó para esta comuna.`;
          bestComunaInfo.style.display = 'block';
        }

        // Fin: redirigir a /resultados igual que antes
        if (status === 'completado' || data.has_results) {
          if (finalMsg) finalMsg.style.display = 'block';
          const finalUrl = runId
            ? `${resultsUrl}${resultsUrl.includes('?') ? '&' : '?'}run_id=${encodeURIComponent(runId)}`
            : resultsUrl;
          setTimeout(() => { window.location.href = finalUrl; }, 1000);
          return;
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
      setTimeout(poll, 1000);
    }

    poll();
  });
})();
