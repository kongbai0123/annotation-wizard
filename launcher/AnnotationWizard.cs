using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Windows.Forms;

internal static class AnnotationWizard
{
    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        bool smoke = args.Length == 1 && args[0] == "--smoke-ui";
        bool first;
        string instance = smoke ? "Smoke" : "Desktop";
        using (var mutex = new Mutex(true, @"Local\AnnotationWizard.60e751f4-8f6d-4d77-91b7-5db028fd630f." + instance, out first))
        {
            if (!first)
            {
                MessageBox.Show("標註精靈已開啟。", "標註精靈", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return 0;
            }
            try
            {
                string root = AppDomain.CurrentDomain.BaseDirectory;
                string python = Path.Combine(root, @".venv\Scripts\pythonw.exe");
                string script = Path.Combine(root, "app.py");
                if (!File.Exists(python) || !File.Exists(script))
                    throw new FileNotFoundException("找不到應用程式檔案。請在標註精靈資料夾啟動，或執行 setup.ps1 重建環境。");

                var start = new ProcessStartInfo(python, "\"" + script + "\"" + (smoke ? " --smoke-ui" : ""))
                {
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WorkingDirectory = root
                };
                start.EnvironmentVariables["PYTHONUTF8"] = "1";
                start.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
                start.EnvironmentVariables["HF_HUB_OFFLINE"] = "1";
                start.EnvironmentVariables["HF_HUB_DISABLE_TELEMETRY"] = "1";
                using (var child = Process.Start(start))
                {
                    child.WaitForExit();
                    if (child.ExitCode != 0)
                        throw new InvalidOperationException("標註精靈啟動失敗。請用 .venv\\Scripts\\python.exe app.py 查看錯誤。");
                }
                return 0;
            }
            catch (Exception error)
            {
                if (!smoke)
                    MessageBox.Show(error.Message, "標註精靈", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
        }
    }
}
