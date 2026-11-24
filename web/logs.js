const allLogs = document.querySelector('#all');
const flaggedLogs = document.querySelector('#flagged');
const detail = document.querySelector('#detail');
const status = document.querySelector('#status');
const refresh = document.querySelector('#refresh');
let currentDetail = 0;

async function openLog(id) {
  const revision = ++currentDetail;
  status.textContent = '';
  try {
    const response = await fetch('/data/' + id);
    if (!response.ok) throw new Error('log not found');
    const log = await response.json();
    if (revision === currentDetail) {
      detail.textContent = '로그 ' + log.id + ' · 점수 ' + log.score.toFixed(2) + '\n\n' + log.input;
    }
  } catch {
    if (revision === currentDetail) status.textContent = '로그를 열 수 없습니다.';
  }
}

function row(log) {
  const item = document.createElement('li');
  const button = document.createElement('button');
  button.textContent = '로그 ' + log.id + ' · ' + log.score.toFixed(2);
  button.addEventListener('click', () => openLog(log.id));
  item.append(button);
  return item;
}

async function loadLogs() {
  refresh.disabled = true;
  status.textContent = '';
  try {
    const response = await fetch('/data');
    if (!response.ok) throw new Error('could not load logs');
    const logs = await response.json();
    allLogs.replaceChildren(...logs.map(row));
    flaggedLogs.replaceChildren(...logs.filter(log => log.flagged).map(row));
    if (!logs.length) status.textContent = '저장된 요청이 없습니다.';
  } catch {
    status.textContent = '로그 목록을 불러올 수 없습니다.';
  } finally {
    refresh.disabled = false;
  }
}

refresh.addEventListener('click', loadLogs);
loadLogs();
