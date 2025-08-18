<script>
  import { onMount } from 'svelte';
  // SSR-safe: load animejs dynamically on the client
  /** @type {any} */
  let animeLib;

  // 親コンポーネントからトラック情報を受け取る (タイトル/シリーズ/作曲者/信頼度)
  export let track = {
    title: 'ココナッツモール',
    series: 'Mario Kart',
    composer: '',
    confidence: 0
  };

  // アニメーションを適用するHTML要素を束縛するための変数
  let titleElement;

  // アニメーションを再生する関数
  function playAnimation() {
    if (!titleElement || !animeLib) return;

    // まずは要素を初期状態（左側で見えない位置）に戻す
    animeLib.set(titleElement, {
      translateX: -250,
      opacity: 0,
    });

    // アニメーションのシーケンスを定義
    animeLib({
      targets: titleElement,
      translateX: 0, // 最終的な位置
      opacity: 1, // 最終的な不透明度
      duration: 800, // アニメーション時間 (ms)
      easing: 'easeOutExpo', // 動きの緩急
    });
  }

  // コンポーネントが最初に表示された時にアニメーションを再生
  onMount(async () => {
    // Load animejs on client only (handle both ESM and CJS shapes)
    const pkg = 'animejs';
    const mod = await import(/* @vite-ignore */ pkg);
    animeLib = (mod && /** @type {any} */ (mod).default) ? /** @type {any} */ (mod).default : mod;
    playAnimation();
  });

  // track.title の値が変更されるたびに、この中のコードが実行される (Svelteのリアクティビティ機能)
  $: if (track && track.title) {
    playAnimation();
  }
</script>

<div class="music-display-container">
  <div class="music-card">
    <h1 bind:this={titleElement} class="track-title">{track.title}</h1>
    <div class="meta-row">
      {#if track.series}
        <span class="badge series">{track.series}</span>
      {/if}
      {#if track.composer}
        <span class="dot">•</span>
        <span class="composer">{track.composer}</span>
      {/if}
      {#if typeof track.confidence === 'number' && track.confidence > 0}
        <span class="dot">•</span>
        <span class="confidence">{Math.round(track.confidence * 100)}%</span>
      {/if}
    </div>
  </div>
</div>

<style>
  .music-display-container {
    /* OBSで表示する際の土台となるスタイル */
    position: fixed;
    bottom: 50px;
    left: 50px;
    /* 必要に応じてフォントなどを読み込む */
    font-family: 'Helvetica', 'Arial', sans-serif;
    color: white;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
  }

  .music-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 14px;
    padding: 12px 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 0 40px rgba(255,255,255,0.04);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
  }

  .track-title {
    font-size: 36px;
    margin: 0 0 6px 0;
    letter-spacing: 0.3px;
    text-shadow: 0 2px 12px rgba(0, 0, 0, 0.55);
    /* アニメーションの初期状態で非表示にしておく */
    opacity: 0;
  }

  .meta-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    color: rgba(255,255,255,0.9);
  }

  .badge.series {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    padding: 2px 8px;
    border-radius: 999px;
    font-weight: 600;
    letter-spacing: 0.2px;
  }

  .composer {
    opacity: 0.95;
  }

  .confidence {
    color: #9be58c;
    font-variant-numeric: tabular-nums;
  }

  .dot { opacity: 0.6; }
</style>