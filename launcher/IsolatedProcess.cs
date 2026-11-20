using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public static class IsolatedProcess
{
    [StructLayout(LayoutKind.Sequential)]
    private struct SecurityCapabilities
    {
        public IntPtr AppContainerSid;
        public IntPtr Capabilities;
        public uint CapabilityCount;
        public uint Reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct StartupInfo
    {
        public uint Size;
        public IntPtr Reserved, Desktop, Title;
        public uint X, Y, XSize, YSize, XChars, YChars, FillAttribute, Flags;
        public ushort ShowWindow, Reserved2Count;
        public IntPtr Reserved2, Input, Output, Error;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct StartupInfoEx
    {
        public StartupInfo Info;
        public IntPtr Attributes;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessInformation
    {
        public IntPtr Process, Thread;
        public uint ProcessId, ThreadId;
    }

    [DllImport("userenv.dll", CharSet = CharSet.Unicode)]
    private static extern int CreateAppContainerProfile(string name, string displayName, string description,
        IntPtr capabilities, uint count, out IntPtr sid);

    [DllImport("userenv.dll", CharSet = CharSet.Unicode)]
    private static extern int DeriveAppContainerSidFromAppContainerName(string name, out IntPtr sid);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern IntPtr FreeSid(IntPtr sid);

    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool ConvertSidToStringSid(IntPtr sid, out IntPtr text);

    [DllImport("kernel32.dll")]
    private static extern IntPtr LocalFree(IntPtr memory);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool InitializeProcThreadAttributeList(IntPtr list, int count, uint flags, ref IntPtr size);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool UpdateProcThreadAttribute(IntPtr list, uint flags, IntPtr attribute, IntPtr value,
        IntPtr valueSize, IntPtr previous, IntPtr returnSize);

    [DllImport("kernel32.dll")]
    private static extern void DeleteProcThreadAttributeList(IntPtr list);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool CreateProcess(string application, StringBuilder command, IntPtr processSecurity,
        IntPtr threadSecurity, bool inheritHandles, uint flags, IntPtr environment, string directory,
        ref StartupInfoEx startup, out ProcessInformation process);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool OpenProcessToken(IntPtr process, uint access, out IntPtr token);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool GetTokenInformation(IntPtr token, int informationClass, IntPtr data,
        uint capacity, out uint actualSize);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(IntPtr thread);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetExitCodeProcess(IntPtr process, out uint status);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateProcess(IntPtr process, uint status);

    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);

    private static void Check(bool ok, string operation)
    {
        if (!ok)
            throw new Win32Exception(Marshal.GetLastWin32Error(), operation);
    }

    private static IntPtr Profile(string name)
    {
        IntPtr sid;
        int result = CreateAppContainerProfile(name, name, "Offline annotation experiment", IntPtr.Zero, 0, out sid);
        if (result == unchecked((int)0x800700b7))
            result = DeriveAppContainerSidFromAppContainerName(name, out sid);
        Marshal.ThrowExceptionForHR(result);
        return sid;
    }

    public static string ProfileSid(string name)
    {
        IntPtr sid = Profile(name);
        try
        {
            IntPtr text;
            Check(ConvertSidToStringSid(sid, out text), "ConvertSidToStringSid");
            try { return Marshal.PtrToStringUni(text); }
            finally { LocalFree(text); }
        }
        finally { FreeSid(sid); }
    }

    private static void VerifyToken(IntPtr process)
    {
        IntPtr token;
        Check(OpenProcessToken(process, 0x0008, out token), "OpenProcessToken");
        IntPtr data = Marshal.AllocHGlobal(4096);
        try
        {
            uint size;
            Check(GetTokenInformation(token, 29, data, 4096, out size), "TokenIsAppContainer");
            if (Marshal.ReadInt32(data) != 1)
                throw new InvalidOperationException("Child is outside the AppContainer.");
            Check(GetTokenInformation(token, 30, data, 4096, out size), "TokenCapabilities");
            if (Marshal.ReadInt32(data) != 0)
                throw new InvalidOperationException("AppContainer unexpectedly has capabilities.");
        }
        finally
        {
            Marshal.FreeHGlobal(data);
            CloseHandle(token);
        }
    }

    public static int Run(string name, string executable, string command, string directory, int timeoutSeconds)
    {
        IntPtr sid = Profile(name);
        IntPtr attrSize = IntPtr.Zero;
        InitializeProcThreadAttributeList(IntPtr.Zero, 1, 0, ref attrSize);
        IntPtr attributes = Marshal.AllocHGlobal(attrSize);
        IntPtr capabilities = Marshal.AllocHGlobal(Marshal.SizeOf<SecurityCapabilities>());
        bool initialized = false;
        ProcessInformation process = new ProcessInformation();
        bool ended = false;
        try
        {
            Check(InitializeProcThreadAttributeList(attributes, 1, 0, ref attrSize), "InitializeProcThreadAttributeList");
            initialized = true;
            Marshal.StructureToPtr(new SecurityCapabilities { AppContainerSid = sid }, capabilities, false);
            Check(UpdateProcThreadAttribute(attributes, 0, (IntPtr)0x00020009, capabilities,
                (IntPtr)Marshal.SizeOf<SecurityCapabilities>(), IntPtr.Zero, IntPtr.Zero), "UpdateProcThreadAttribute");
            StartupInfoEx startup = new StartupInfoEx();
            startup.Info.Size = (uint)Marshal.SizeOf<StartupInfoEx>();
            startup.Attributes = attributes;
            const uint flags = 0x00080000 | 0x00000004 | 0x08000000;
            Check(CreateProcess(executable, new StringBuilder(command), IntPtr.Zero, IntPtr.Zero, false, flags,
                IntPtr.Zero, directory, ref startup, out process), "CreateProcess");
            VerifyToken(process.Process);
            Check(ResumeThread(process.Thread) != uint.MaxValue, "ResumeThread");
            uint elapsed = WaitForSingleObject(process.Process, checked((uint)timeoutSeconds * 1000));
            if (elapsed == 258)
                throw new TimeoutException("AppContainer child exceeded the requested runtime.");
            Check(elapsed == 0, "WaitForSingleObject");
            ended = true;
            uint status;
            Check(GetExitCodeProcess(process.Process, out status), "GetExitCodeProcess");
            return unchecked((int)status);
        }
        finally
        {
            if (process.Process != IntPtr.Zero)
            {
                if (!ended)
                    TerminateProcess(process.Process, 1);
                CloseHandle(process.Process);
                CloseHandle(process.Thread);
            }
            if (initialized)
                DeleteProcThreadAttributeList(attributes);
            Marshal.FreeHGlobal(attributes);
            Marshal.FreeHGlobal(capabilities);
            FreeSid(sid);
        }
    }
}
