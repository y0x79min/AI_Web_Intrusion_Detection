const form = document.querySelector('#request');
const status = document.querySelector('#status');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = form.querySelector('button');
  button.disabled = true;
  status.textContent = '분석 중';
  try {
    const input = new FormData(form).get('input');
    const response = await fetch('/demo/request', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input }),
    });
    if (!response.ok) throw new Error('request failed');
    const log = await response.json();
    const state = log.flagged ? '의심' : '일반';
    status.textContent = '로그 ' + log.id + ' · 점수 ' + log.score.toFixed(2) + ' · ' + state;
  } catch {
    status.textContent = '요청을 분석할 수 없습니다.';
  } finally {
    button.disabled = false;
  }
});
