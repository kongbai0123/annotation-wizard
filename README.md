# 標註精靈

Windows 偵測框標註工具。雙擊 `AnnotationWizard.exe` 開啟繁體中文介面。程式、模型和執行環境都在同一個資料夾，從桌面啟動時可為執行檔建立捷徑。

1. 按「新建專案」，選一個新資料夾，匯入要標註的圖片並新增類別。
2. 在「偵測框」模式選好類別，用左鍵從物件的一角拖到對角。
3. 按「儲存框」或 Enter。要調整已存的框，選取後按「修改所選框」，重新框選後儲存回同一筆標註。
4. 匯出 YOLO 偵測標註會產生 `images/`、`labels/` 與 `classes.txt`。每個 YOLO 標註記錄類別編號與相對於原圖的中心、寬、高。匯出後可依資料來源安排訓練、驗證與測試分割。

需要精細輪廓時，切換「精細遮罩」。可用框和前景／背景點請 SAM 2.1 產生候選，確認後儲存遮罩。

「自動修整」可選平衡、輕微或關閉。平衡會平滑細小毛邊、移除太小的離散區塊、填補小孔，保留提示點和較大的開口。「看原始遮罩」可查看修整前的結果。

要修改已存遮罩，先在清單選取，再按「修整遮罩」。用「填洞」點一下封閉缺口，或用「移除孤島」點選多餘的離散區塊。「加畫」和「擦除」可修長段輪廓，改錯可按「撤回修整」。確認後按「儲存修整」。遮罩有修改時，該圖片會回到待審核。

每個候選對應一個物件，類別由操作者指定。推論在本機執行。若沒有可用的 NVIDIA GPU，SAM 2.1 可用 CPU 執行。對前景和背景反差明顯的圖片，可切換為 GrabCut。

專案資料夾保存 `project.json` 與轉正方向的圖片副本。YOLO 與 COCO 匯出包含已儲存的標註，以及經操作者確認沒有目標的圖片。YOLO 偵測匯出會取精細遮罩的外接矩形，COCO 匯出則保留原有的框或遮罩。尚未標註且未核對的圖片不列入匯出。

## 影片自動標註

若要自己選用文字候選框，先在影片預覽勾選「只匯入關鍵畫格」。預設每秒擷取一張，可調，這個模式可直接使用本機影片。

1. 開啟專案後按「影片自動標註」，選擇要審查的原始影片。可看不同時間點的畫面，或按「播放原片」核對拍攝是否連續。
2. 選擇你信任、類別符合該影片的本機 YOLO Detect `.pt` 模型，設定要保存的畫格率和信心門檻。審查完成再按「確認影片並匯入」。審查階段只讀取影片和模型檔案，確認後會再次核對檔案雜湊，再啟動推論。
3. 程式逐幀執行 YOLO 偵測和 ByteTrack，再依設定儲存取樣畫格。影格保留原片資訊、時間；候選框保留類別、信心分數和可用的追蹤 ID。同一影片的同一畫格重複匯入時保留原標註。
4. 開啟「影片待審」畫格檢查漏標與誤標。按「標記畫格已審核」後該畫格才可匯出，空畫格也需要明確核對。日後製作訓練與測試集時，把同一原片的畫格放在同一個分割。

影片標註會使用本機已安裝的 Ultralytics、PyTorch 和 LAP。可透過環境變數 `ANNOTATION_WIZARD_VIDEO_PYTHON` 指定影片 Python 路徑。目前亦可使用同層 Vision Workbench 既有的 Ultralytics 執行環境。如需重建專案內的 LAP 依賴，使用相同的影片 Python 執行以下指令。

```powershell
$VideoPython = 'D:\software\vision-workbench\.venv-models\ultralytics\Scripts\python.exe'
& $VideoPython -m pip install --only-binary=:all: --no-deps --target .video-deps lap==0.5.13
```

## 文字候選框

在「偵測框」模式輸入物件名稱，例如 `car door`，按「找框」。從最多五個候選中選擇預覽，可拖四角微調。選好類別後按「儲存框」，才會成為標註。

程式使用同層獨立的 `MM-Grounding-DINO-Tiny` 資料夾，也可用 `ANNOTATION_WIZARD_DINO_DIR` 指定本機路徑。需要官方 O365＋GoldG 快照（修訂 `80b3081e63ff7c265a9b404296d848f0f0aa0006`）及本機 `health/manifest.json` 對照集。對照集為 JSON 陣列，記錄 `file`、`sha256`、`prompt`（`person` 或 `car`）和原圖像素座標 `xyxy`，圖片放在同一個 `health` 資料夾。權重、輸出頭及對照推論通過驗證後才顯示候選。模型與測試圖由本機載入。

## 重建執行環境

需要 Windows、64 位元 Python 3.13，以及 Windows .NET Framework 4.x 的 C# 編譯器。安裝程式會準備專案自己的 `.venv`、驗證本機權重，並重新產生 `AnnotationWizard.exe`。

```powershell
Set-Location D:\software\annotation-wizard
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

CPU 主機可加 `-CpuOnly`。已有下載的模型資料夾可加 `-ModelSource <路徑>`，安裝會驗證來源檔案的 SHA-256。[Meta 的 SAM 2.1 Tiny 模型](https://huggingface.co/facebook/sam2.1-hiera-tiny/tree/de431c4043854a71d8101e17995dfe596bf101a5)
