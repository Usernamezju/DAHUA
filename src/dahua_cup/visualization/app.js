fetch("/api/system")
  .then((response) => response.ok ? response.json() : Promise.reject(response.statusText))
  .then((value) => { document.querySelector("#system").textContent = JSON.stringify(value, null, 2); })
  .catch((error) => { document.querySelector("#system").textContent = `服务不可用：${error}`; });
