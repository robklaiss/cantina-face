const API_BASE = "";

const state = {
  token: localStorage.getItem("parent_token") || null,
  user: null,
  students: [],
  topups: [],
  scheduledOrders: [],
  menus: [],
  linkRequests: [],
  products: [],
  menuSelections: [],
  loading: false,
  currentRoute: "dashboard",
};

const selectors = {
  app: () => document.getElementById("app"),
};

const routes = {
  dashboard: () => renderDashboard(),
  topups: () => renderTopUps(),
  orders: () => renderOrders(),
  menu: () => renderMenu(),
};

function setToken(token) {
  state.token = token;
  if (token) {
    localStorage.setItem("parent_token", token);
  } else {
    localStorage.removeItem("parent_token");
  }
}

async function api(path, options = {}) {
  const headers = options.headers || {};
  if (state.token) {
    headers["Authorization"] = `Bearer ${state.token}`;
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...headers,
    },
  });
  if (response.status === 401) {
    setToken(null);
    renderLogin("Sesión expirada");
    throw new Error("Sesión expirada");
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || "Error inesperado");
  }
  return response.json();
}

async function boot() {
  if (!state.token) {
    renderLogin();
    return;
  }
  try {
    state.loading = true;
    navigate(state.currentRoute);
    await Promise.all([
      loadCurrentUser(),
      loadStudents(),
      loadTopUps(),
      loadScheduledOrders(),
      loadMenus(),
      loadLinkRequests(),
      loadProducts(),
      loadMenuSelections(),
    ]);
  } catch (err) {
    console.error(err);
    alert(err.message);
    setToken(null);
    renderLogin(err.message);
    return;
  } finally {
    state.loading = false;
  }
  navigate(state.currentRoute || "dashboard");
}

async function loadCurrentUser() {
  state.user = await api("/auth/me");
}

async function loadStudents() {
  state.students = await api("/api/parents/students");
}

async function loadTopUps() {
  state.topups = await api("/api/parents/topups");
}

async function loadScheduledOrders() {
  state.scheduledOrders = await api("/api/parents/scheduled-orders");
}

async function loadMenus() {
  const today = new Date().toISOString().split("T")[0];
  state.menus = await api(`/api/daily-menu?start=${today}`);
}

async function loadLinkRequests() {
  state.linkRequests = await api("/api/parents/link-requests");
}

async function loadProducts() {
  state.products = await api("/api/products");
}

async function loadMenuSelections() {
  state.menuSelections = await api("/api/parents/menu-selections");
}

function render(template) {
  selectors.app().innerHTML = template;
}

function renderLogin(errorMessage = null) {
  render(`
    <div class="auth-card">
      <h1>Ingreso Padres</h1>
      ${errorMessage ? `<p class="error">${errorMessage}</p>` : ""}
      <form id="login-form">
        <label>Email
          <input type="email" name="email" required />
        </label>
        <label>Contraseña
          <input type="password" name="password" required />
        </label>
        <button type="submit">Ingresar</button>
      </form>
      <p class="note">¿Aún no tienes cuenta? Solicita acceso en la institución.</p>
    </div>
  `);

  document.getElementById("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const body = new URLSearchParams();
    body.append("username", formData.get("email"));
    body.append("password", formData.get("password"));
    try {
      const response = await fetch("/auth/token", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body,
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || "Credenciales inválidas");
      }
      const data = await response.json();
      setToken(data.access_token);
      boot();
    } catch (err) {
      renderLogin(err.message);
    }
  });
}

function renderShell(mainContent) {
  render(`
    <header class="app-header">
      <div class="brand-block">
        <img src="/siloe-logo-blanco.png" alt="Siloe" class="brand-logo" />
        <div>
          <p class="welcome">Hola, ${state.user?.first_name || state.user?.full_name || "familia"}</p>
          <h1>Portal de padres</h1>
        </div>
      </div>
      <nav>
        ${navButton("dashboard", "Inicio")}
        ${navButton("topups", "Saldos")}
        ${navButton("orders", "Pedidos")}
        ${navButton("menu", "Menú")}
        <button id="logout">Salir</button>
      </nav>
    </header>
    <main>
      ${mainContent}
    </main>
  `);
  bindNavigation();
}

function navButton(route, label) {
  const active = state.currentRoute === route ? "active" : "";
  return `<button data-route="${route}" class="${active}">${label}</button>`;
}

function bindNavigation() {
  document.getElementById("logout")?.addEventListener("click", () => {
    setToken(null);
    renderLogin();
  });
  document.querySelectorAll("button[data-route]").forEach((btn) => {
    btn.addEventListener("click", () => navigate(btn.dataset.route));
  });
}

function navigate(route) {
  state.currentRoute = route || "dashboard";
  const view = routes[state.currentRoute] || routes.dashboard;
  view();
}

function renderDashboard() {
  const studentsHtml = state.students
    .map(
      (student) => `
        <article class="card">
          <header>
            <h2>${student.name}</h2>
            <span>${student.grade}</span>
          </header>
          <p class="balance">Saldo: <strong>${formatGs(student.balance)}</strong></p>
          <div class="actions">
            <button data-route="topups">Cargar saldo</button>
            <button data-route="orders">Pedidos</button>
          </div>
        </article>
      `
    )
    .join("");

  const upcomingOrders = state.scheduledOrders
    .slice(0, 4)
    .map(
      (order) => `
        <li>
          <div>
            <strong>${formatDate(order.scheduled_for)}</strong>
            <span>${findStudentName(order.student_id)}</span>
          </div>
          <small>${order.items.map((item) => `${item.quantity}x ${item.product_name || "Producto"}`).join(", ")}</small>
        </li>
      `
    )
    .join("");

  const linkRequestsHtml = state.linkRequests
    .map(
      (request) => `
        <li>
          <div>
            <strong>${request.student_name}</strong>
            <small>${request.student_grade || ""}</small>
          </div>
          <span class="chip status-${request.status}">${humanizeStatus(request.status)}</span>
          ${request.admin_notes ? `<small>${request.admin_notes}</small>` : ""}
        </li>
      `
    )
    .join("");

  renderShell(`
    <section>
      <h2>Hijos vinculados</h2>
      <div class="grid">${studentsHtml || "<p>No hay estudiantes asignados.</p>"}</div>
    </section>
    <section>
      <h2>Próximos pedidos</h2>
      <ul class="list">${upcomingOrders || "<li>Sin pedidos programados</li>"}</ul>
    </section>
    <section class="link-section">
      <div>
        <h2>Solicitar vinculación</h2>
        <p class="note">Completá los datos del alumno para que el colegio valide el vínculo.</p>
        <form id="link-form" class="stack">
          <label>Nombre y apellido del alumno
            <input type="text" name="student_name" required />
          </label>
          <label>Grado / Curso
            <input type="text" name="student_grade" />
          </label>
          <label>Código / Cédula del alumno
            <input type="text" name="student_identifier" />
          </label>
          <label>Notas adicionales
            <textarea name="notes" rows="3"></textarea>
          </label>
          <button type="submit">Enviar solicitud</button>
        </form>
      </div>
      <div>
        <h2>Solicitudes recientes</h2>
        <ul class="list">${linkRequestsHtml || "<li>Aún no realizaste solicitudes.</li>"}</ul>
      </div>
    </section>
  `);

  const linkForm = document.getElementById("link-form");
  if (linkForm) {
    linkForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const formData = new FormData(linkForm);
      const payload = {
        student_name: formData.get("student_name"),
        student_grade: formData.get("student_grade") || null,
        student_identifier: formData.get("student_identifier") || null,
        notes: formData.get("notes") || null,
      };
      try {
        linkForm.classList.add("loading");
        await api("/api/parents/link-requests", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        await loadLinkRequests();
        alert("Solicitud enviada. Será revisada por la institución.");
        linkForm.reset();
        renderDashboard();
      } catch (err) {
        alert(err.message);
      } finally {
        linkForm.classList.remove("loading");
      }
    });
  }
}

function renderTopUps() {
  if (!state.students.length) {
    renderShell('<section><p class="note">Aún no tienes hijos vinculados. Solicita el vínculo primero.</p></section>');
    return;
  }

  const balanceCards = state.students
    .map(
      (student) => `
        <article class="mini-card">
          <p class="label">${student.name}</p>
          <strong>${formatGs(student.balance)}</strong>
          <small>${student.grade || ''}</small>
        </article>
      `
    )
    .join(" ");

  const requestsHtml = state.topups
    .map((topup) => `
      <li>
        <div>
          <strong>${formatGs(topup.total_amount)}</strong>
          <span class="chip status-${topup.status}">${humanizeStatus(topup.status)}</span>
        </div>
        <small>${formatDateTime(topup.created_at)}</small>
        <p class="note">Modo: ${topup.allocation_mode === "equal" ? "Distribución equitativa" : "Asignación personalizada"}</p>
        ${renderAllocations(topup.allocations)}
      </li>
    `)
    .join("");

  const studentInputs = state.students
    .map(
      (student) => `
        <label>${student.name}
          <input type="number" min="0" step="1000" data-student-alloc="${student.id}" placeholder="0" />
        </label>
      `
    )
    .join("");

  renderShell(`
    <section class="wide-section">
      <h2>Saldo actual por hijo</h2>
      <div class="mini-grid">${balanceCards}</div>
    </section>
    <section>
      <h2>Solicitar carga de saldo</h2>
      <form id="topup-form" class="stack">
        <label>Monto total
          <input type="number" name="total_amount" min="1000" step="500" required />
        </label>
        <label>Modo de asignación
          <select name="allocation_mode" id="allocation-mode">
            <option value="equal">Distribuir por igual</option>
            <option value="custom">Personalizado por alumno</option>
          </select>
        </label>
        <div id="custom-allocations" class="hidden">
          <p class="note">Define cuánto recibirá cada hijo (la suma debe coincidir con el monto total).</p>
          <div class="form-grid">${studentInputs}</div>
        </div>
        <label>Referencia de pago / comprobante
          <input type="text" name="payment_reference" placeholder="Ej: Transferencia N°123" />
        </label>
        <button type="submit">Enviar solicitud</button>
      </form>
    </section>
    <section>
      <h2>Solicitudes anteriores</h2>
      <ul class="list">${requestsHtml || "<li>Aún no solicitaste créditos.</li>"}</ul>
    </section>
  `);

  const form = document.getElementById("topup-form");
  const modeSelect = document.getElementById("allocation-mode");
  const allocationsBlock = document.getElementById("custom-allocations");
  modeSelect.addEventListener("change", () => {
    allocationsBlock.classList.toggle("hidden", modeSelect.value !== "custom");
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    const payload = {
      total_amount: Number(formData.get("total_amount")),
      allocation_mode: formData.get("allocation_mode"),
      payment_reference: formData.get("payment_reference") || null,
    };

    if (payload.allocation_mode === "custom") {
      const perStudent = {};
      document.querySelectorAll("[data-student-alloc]").forEach((input) => {
        const value = Number(input.value || "0");
        perStudent[input.dataset.studentAlloc] = value;
      });
      payload.per_student_amounts = perStudent;
    }

    try {
      form.classList.add("loading");
      await api("/api/parents/topups", { method: "POST", body: JSON.stringify(payload) });
      await Promise.all([loadTopUps(), loadStudents()]);
      alert("Solicitud enviada. Se acreditará cuando el colegio confirme el pago.");
      navigate("topups");
    } catch (err) {
      alert(err.message);
    } finally {
      form.classList.remove("loading");
    }
  });
}

function renderOrders() {
  if (!state.students.length) {
    renderShell('<section><p class="note">Necesitas al menos un hijo vinculado para programar pedidos.</p></section>');
    return;
  }
  if (!state.products.length) {
    renderShell('<section><p class="note">No hay productos configurados todavía.</p></section>');
    return;
  }

  const ordersHtml = state.scheduledOrders
    .map(
      (order) => `
        <li>
          <div>
            <strong>${findStudentName(order.student_id)}</strong>
            <small>${formatDate(order.scheduled_for)}</small>
          </div>
          <span class="chip status-${order.status}">${humanizeStatus(order.status)}</span>
          <small>${order.pay_from_balance ? "Se descontará del saldo" : "Pago en cantina"}</small>
          <small>${order.items.map((item) => `${item.quantity}x ${item.product_name || "Producto"}`).join(", ")}</small>
        </li>
      `
    )
    .join("");

  renderShell(`
    <section>
      <h2>Programar pedido</h2>
      <form id="order-form" class="stack">
        <label>Alumno
          <select name="student_id" required>
            <option value="">Selecciona un hijo</option>
            ${state.students.map((s) => `<option value="${s.id}">${s.name} (${s.grade})</option>`).join("")}
          </select>
        </label>
        <label>Fecha de entrega
          <input type="date" name="scheduled_for" id="scheduled-for" required />
        </label>
        <label>Notas para la cantina
          <textarea name="notes" rows="2" placeholder="Opcional"></textarea>
        </label>
        <label>¿Cómo querés pagar?
          <select name="pay_from_balance" required>
            <option value="true" selected>Descontar del saldo del alumno</option>
            <option value="false">Lo pagaré aparte en la cantina</option>
          </select>
        </label>
        <div>
          <p class="note">Agrega uno o más productos:</p>
          <div id="order-items" class="order-items"></div>
          <button type="button" class="ghost-btn" id="add-order-item">+ Producto</button>
        </div>
        <button type="submit">Guardar pedido</button>
      </form>
    </section>
    <section>
      <h2>Pedidos programados</h2>
      <ul class="list">${ordersHtml || "<li>No tienes pedidos programados.</li>"}</ul>
    </section>
  `);

  const dateInput = document.getElementById("scheduled-for");
  if (dateInput) {
    dateInput.min = new Date().toISOString().split("T")[0];
  }

  const itemsContainer = document.getElementById("order-items");
  const addItemBtn = document.getElementById("add-order-item");
  const addRow = () => addOrderItemRow(itemsContainer);
  addItemBtn.addEventListener("click", addRow);
  itemsContainer.addEventListener("click", (event) => {
    if (event.target.matches(".remove-item")) {
      event.target.closest(".order-item-row")?.remove();
    }
  });
  addRow();

  document.getElementById("order-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    const items = collectOrderItems(itemsContainer);
    if (!items.length) {
      alert("Agrega al menos un producto");
      return;
    }
    const payload = {
      student_id: formData.get("student_id"),
      scheduled_for: formData.get("scheduled_for"),
      notes: formData.get("notes") || null,
      pay_from_balance: formData.get("pay_from_balance") !== "false",
      items,
    };
    try {
      form.classList.add("loading");
      await api("/api/parents/scheduled-orders", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      await loadScheduledOrders();
      alert("Pedido programado correctamente");
      navigate("orders");
    } catch (err) {
      alert(err.message);
    } finally {
      form.classList.remove("loading");
    }
  });
}

function renderMenu() {
  const menuCards = state.menus
    .map(
      (menu) => `
        <article class="menu-card">
          <header>
            <h3>${formatDate(menu.menu_date)}</h3>
            <p>${menu.title || "Menú"}</p>
          </header>
          <ul class="menu-items">
            ${menu.items
              .map((item) => `
                <li>
                  <div>
                    <strong>${item.name}</strong>
                    <small>${item.meal_type || ""}</small>
                  </div>
                  ${state.students.length
                    ? `
                        <form class="menu-select-form" data-menu="${menu.id}" data-item="${item.id}">
                          <select name="student_id" required>
                            <option value="">Selecciona hijo</option>
                            ${state.students.map((s) => `<option value="${s.id}">${s.name}</option>`).join("")}
                          </select>
                          <button type="submit">Asignar</button>
                        </form>
                      `
                    : '<small class="note">Vincula un hijo para asignar platos.</small>'}
                </li>
              `)
              .join("")}
          </ul>
        </article>
      `
    )
    .join("");

  const selectionsHtml = state.menuSelections
    .map(
      (selection) => `
        <li>
          <div>
            <strong>${findStudentName(selection.student_id)}</strong>
            <small>${formatDate(selection.menu_date)}</small>
          </div>
          <span>${findMenuItemName(selection.menu_item_id)}</span>
        </li>
      `
    )
    .join("");

  renderShell(`
    <section>
      <h2>Menú del día</h2>
      ${menuCards || '<p class="note">Aún no hay menús publicados.</p>'}
    </section>
    <section>
      <h2>Selecciones recientes</h2>
      <ul class="list">${selectionsHtml || "<li>No registraste selecciones.</li>"}</ul>
    </section>
  `);

  document.querySelectorAll(".menu-select-form").forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const formData = new FormData(form);
      const menuId = form.dataset.menu;
      const menuItemId = Number(form.dataset.item);
      const studentId = formData.get("student_id");
      try {
        form.classList.add("loading");
        await api(`/api/daily-menu/${menuId}/selections`, {
          method: "POST",
          body: JSON.stringify({
            menu_item_id: menuItemId,
            student_id: studentId,
            notes: null,
          }),
        });
        await loadMenuSelections();
        alert("Selección guardada");
        navigate("menu");
      } catch (err) {
        alert(err.message);
      } finally {
        form.classList.remove("loading");
      }
    });
  });
}

function renderAllocations(allocations = {}) {
  const entries = Object.entries(allocations || {});
  if (!entries.length) return "";
  const rows = entries
    .map(([studentId, amount]) => `<li>${findStudentName(studentId)}: ${formatGs(amount)}</li>`)
    .join("");
  return `<ul class="list allocations-list">${rows}</ul>`;
}

function addOrderItemRow(container) {
  const row = document.createElement("div");
  row.className = "order-item-row";
  row.innerHTML = `
    <select name="product_id" required>
      <option value="">Producto</option>
      ${state.products.map((p) => `<option value="${p.id}">${p.name} (${formatGs(p.price)})</option>`).join("")}
    </select>
    <input type="number" name="quantity" min="1" value="1" required />
    <button type="button" class="remove-item">✕</button>
  `;
  container.appendChild(row);
}

function collectOrderItems(container) {
  const rows = Array.from(container.querySelectorAll(".order-item-row"));
  return rows
    .map((row) => {
      const productId = Number(row.querySelector("select")?.value || 0);
      const quantity = Number(row.querySelector("input")?.value || 0);
      if (!productId || !quantity) return null;
      return { product_id: productId, quantity };
    })
    .filter(Boolean);
}

function findStudentName(studentId) {
  const student = state.students.find((s) => String(s.id) === String(studentId));
  return student ? student.name : "Alumno";
}

function humanizeStatus(status) {
  switch (status) {
    case "approved":
      return "Aprobada";
    case "rejected":
      return "Rechazada";
    case "dispatched":
      return "Entregado";
    case "cancelled":
      return "Cancelado";
    case "pending":
      return "Pendiente";
    default:
      return status || "Pendiente";
  }
}

function formatGs(amount) {
  const value = Number(amount) || 0;
  return value.toLocaleString("es-PY", { maximumFractionDigits: 0 }) + " Gs";
}

function formatDate(dateStr) {
  if (!dateStr) return "";
  const date = new Date(dateStr);
  return date.toLocaleDateString("es-PY", { weekday: "short", day: "numeric", month: "short" });
}

function formatDateTime(dateStr) {
  if (!dateStr) return "";
  const date = new Date(dateStr);
  return date.toLocaleString("es-PY", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function findMenuItemName(menuItemId) {
  for (const menu of state.menus) {
    const item = menu.items.find((i) => Number(i.id) === Number(menuItemId));
    if (item) return item.name;
  }
  return "Menú";
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/parents/service-worker.js").catch((err) => {
      console.warn("SW registration failed", err);
    });
  });
}

boot();
