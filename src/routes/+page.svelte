<script>
  import MusicDisplay from '$lib/MusicDisplay.svelte';
  import { onMount } from 'svelte';

  // デモ用の初期トラック（WS未接続時のフォールバック表示）
  const demoList = [
    { title: 'マリオカートスタジアム', series: 'Mario Kart 8', composer: '', confidence: 0 },
    { title: 'ウォーターパーク', series: 'Mario Kart 8', composer: '', confidence: 0 },
    { title: 'スイーツキャニオン', series: 'Mario Kart 8', composer: '', confidence: 0 },
  ];

  let currentTrack = demoList[0];

  // ローカル認識サービスからのメタデータを受信（例: ws://127.0.0.1:8765）
  onMount(() => {
    /** @type {WebSocket | null} */
    let ws = null;
    /** @type {number | null} */
    let demoTimer = null;

    // デモローテーション（WS接続まで）
    let idx = 0;
    demoTimer = setInterval(() => {
      idx = (idx + 1) % demoList.length;
      currentTrack = demoList[idx];
    }, 5000);

    function connect() {
      try {
        ws = new WebSocket('ws://127.0.0.1:8765');
      } catch (e) {
        // 再試行
        setTimeout(connect, 1500);
        return;
      }

      ws.onopen = () => {
        // 接続できたらデモ停止
        if (demoTimer) {
          clearInterval(demoTimer);
          demoTimer = null;
        }
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          currentTrack = {
            title: msg.title ?? currentTrack.title,
            series: msg.series ?? '',
            composer: msg.composer ?? '',
            confidence: typeof msg.confidence === 'number' ? msg.confidence : 0
          };
        } catch (_) {
          // ignore
        }
      };

      ws.onclose = () => {
        // 再接続
        setTimeout(connect, 1500);
      };

      ws.onerror = () => {
        try { ws && ws.close(); } catch (_) {}
      };
    }

    connect();

    return () => {
      if (demoTimer) clearInterval(demoTimer);
      if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    };
  });

</script>

<MusicDisplay track={currentTrack} />