package json

import (
	"bytes"

	"github.com/bytedance/sonic"
	"github.com/rrabit42/aidrop-core/common/lib/log"
	_sync "github.com/rrabit42/aidrop-core/common/sync"
)

type handler struct {
	marshal    func(val interface{}) ([]byte, error)
	unmarshal  func(buf []byte, val interface{}) error
	memoryPool *_sync.Pool[*bytes.Buffer]
	log        log.Logger
}

var JsonHandler handler

func init() {
	JsonHandler = handler{
		marshal:   sonic.Marshal,
		unmarshal: sonic.Unmarshal,
		memoryPool: _sync.NewPool[*bytes.Buffer](func() *bytes.Buffer {
			return new(bytes.Buffer)
		}),
		log: log.New("common", "json/handler"),
	}
}

func (h handler) Marshal(v interface{}) ([]byte, error) {
	memoryBuf := h.memoryPool.Get()
	defer h.memoryPool.Put(memoryBuf)

	memoryBuf.Reset()
	data, err := h.marshal(v)
	memoryBuf.Write(data)

	if err != nil {
		h.log.Error("Failed to marshal by handler", "err", err)
		return nil, err
	}

	return memoryBuf.Bytes(), nil
}

func (h handler) Unmarshal(buffer []byte, v interface{}) error {
	memoryBuf := h.memoryPool.Get()
	defer h.memoryPool.Put(memoryBuf)

	memoryBuf.Reset()
	memoryBuf.Write(buffer)

	err := h.unmarshal(memoryBuf.Bytes(), v)

	if err != nil {
		h.log.Error("Failed to unmarshal by handler", "buffer", string(buffer), "err", err)
		return err
	}

	return nil
}

func (h handler) Handle(buf interface{}, v interface{}) error {
	data, err := h.Marshal(buf)

	if err != nil {
		return err
	}

	err = h.Unmarshal(data, v)

	if err != nil {
		return err
	}

	return nil
}
