package log

import (
	"os"

	"gopkg.in/natefinch/lumberjack.v2"
)

var (
	root          = &logger{[]interface{}{}, new(swapHandler)}
	StdoutHandler = StreamHandler(os.Stdout, LogfmtFormat())
	StderrHandler = StreamHandler(os.Stderr, LogfmtFormat())
)

func SetRoot(useTerminal bool, verbosityTerminal int, useFile bool, verbosityFile int, filePath string) {
	hs := []Handler{}

	if useTerminal { //터미널 모드
		hs = append(hs, LvlFilterHandler(Lvl(verbosityTerminal),
			StreamHandler(os.Stdout, TerminalFormat(true))))
	}
	if useFile { //파일모드
		hs = append(hs, LvlFilterHandler(Lvl(verbosityFile), StreamHandler(&lumberjack.Logger{
			Filename:   filePath,
			MaxSize:    1024, // megabytes
			MaxBackups: 3,
			MaxAge:     28,   //days
			Compress:   true, // disabled by default
		}, JSONFormat())))
	}
	root.SetHandler(MultiHandler(hs...))

	root.Trace("Log handlers initialized", "module", "log", "terminal", useTerminal, "file", useFile, "path", filePath)
}

func init() {
	root.SetHandler(DiscardHandler())
}

// New returns a new logger with the given context.
// NewModule is a convenient alias for process.New
func NewModule(module string, ctx ...interface{}) Logger {
	logger := root.New("module", module)
	if len(ctx) > 0 {
		return logger.New(ctx)
	}
	return logger
}

// New returns a new logger with the given context.
// New is a convenient alias for Root().New
func New(ctx ...interface{}) Logger {
	return root.New(ctx...)
}

// Root returns the root logger
func Root() Logger {
	return root
}

// The following functions bypass the exported logger methods (logger.Debug,
// etc.) to keep the call depth the same for all paths to logger.write so
// runtime.Caller(2) always refers to the call site in client code.

// Trace is a convenient alias for Root().Trace
func Trace(msg string, ctx ...interface{}) {
	root.write(msg, LvlTrace, ctx, skipLevel)
}

// Debug is a convenient alias for Root().Debug
func Debug(msg string, ctx ...interface{}) {
	root.write(msg, LvlDebug, ctx, skipLevel)
}

// Info is a convenient alias for Root().Info
func Info(msg string, ctx ...interface{}) {
	root.write(msg, LvlInfo, ctx, skipLevel)
}

// Warn is a convenient alias for Root().Warn
func Warn(msg string, ctx ...interface{}) {
	root.write(msg, LvlWarn, ctx, skipLevel)
}

// Error is a convenient alias for Root().Error
func Error(msg string, ctx ...interface{}) {
	root.write(msg, LvlError, ctx, skipLevel)
}

// Crit is a convenient alias for Root().Crit
func Crit(msg string, ctx ...interface{}) {
	root.write(msg, LvlCrit, ctx, skipLevel)
	os.Exit(1)
}

// Output is a convenient alias for write, allowing for the modification of
// the calldepth (number of stack frames to skip).
// calldepth influences the reported line number of the log message.
// A calldepth of zero reports the immediate caller of Output.
// Non-zero calldepth skips as many stack frames.
func Output(msg string, lvl Lvl, calldepth int, ctx ...interface{}) {
	root.write(msg, lvl, ctx, calldepth+skipLevel)
}
