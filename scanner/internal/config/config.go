package config

import (
	"os"

	"github.com/naoina/toml"
)

type Config struct {
	Chain struct {
		Name        string
		URL         string
		Start       int64
		ReadingUnit int64
		Contracts   []string
	}

	Log struct {
		Terminal struct {
			Use       bool
			Verbosity int
		}
		File struct {
			Use       bool
			Verbosity int
			FileName  string
		}
	}
}

func NewConfig(path string) *Config {
	c := new(Config)

	if file, err := os.Open(path); err != nil {
		panic(err)
	} else {
		defer file.Close()
		if err := toml.NewDecoder(file).Decode(c); err != nil {
			panic(err)
		}
		return c
	}
}
