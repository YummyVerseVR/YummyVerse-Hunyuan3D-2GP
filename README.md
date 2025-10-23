# YummyVerse-Hunyuan3D-2GP

## 実行方法
```
uv sync
uv run src/entry.py
```

## 引数一覧
- `--port / -p`: サーバのポート番号
    - デフォルト: 8000

- `--config / -c`: 設定ファイルのパス
    - デバッグ等の用途で設定ファイルを変更したい場合に指定します.
    - デフォルト: `./settings/config.json`

- `--debug / -d`: デバッグモードを有効化
    - デバッグが有効な場合, 一部のAPIエンドポイントが有効化される他, loggingを除くすべてのネットワークを必要とする処理が無効化されます.
    - デフォルト: 無効

- `--logging / -l`: ネットワークロギングを有効化
    - 有効な場合はpylognetを使用してログサーバにログを送信します.
    - デフォルト: 無効

## config.jsonの仕様
設定ファイル`config.json`は以下の形式で記述します.
```json
{
    "hunyuan3d": {
        "max_seed": "乱数シードの最大値 (任意)",
        "supported_formats": ["サポートされている3Dモデルのファイル形式 (任意)"],
        "model": "モデル名 (任意)",
        "subfolder": "モデルのサブフォルダ (任意)",
        "texgen": "texgenモデル名 (任意)",
        "device": "cuda or cpu (任意)",
        "mc_algo": "マルチビュー合成アルゴリズム (任意)",
        "cache_path": "キャッシュパス (任意)",
        "profile": "プロファイル名 (任意)",
        "verbose": "詳細ログを有効化 (任意)",
        "enable_flashvdm": true or false (任意)",
        "disable_tex": true or false (任意)",
        "low_vram_mode": true or false (任意)",
        "compile": true or false (任意)",
        "mini": true or false (任意)",
        "turbo": true or false (任意)",
        "mv": true or false (任意)",
        "h2": true or false (任意)",
        "random_seed": true or false (任意),
        "seed": "乱数シード (任意)",
        "inference_steps": "試行回数 (任意)",
        "guidance_scale": "ガイダンススケール (任意)",
        "octree_resolution": "オクツリー解像度 (任意)",
        "checkbox_rembg": true or false (任意)",
        "num_chunks": "チャンク数 (任意)"
    },
    "endpoints": {
        "control": "コントロールサーバのURL (任意)",
        "logger": "ログサーバのURL (任意)"
    }
}
```

## APIエンドポイント
このサーバは以下のAPIエンドポイントを提供します. 詳細な仕様についてはFastAPIの自動生成ドキュメント`http://0.0.0.0:<port>/docs`を参照してください.
