package utils

import (
	"github.com/rrabit42/aidrop-core/common/json"
)

func ToJSON(t any) (any, error) {
	var v any

	if bytes, err := json.JsonHandler.Marshal(t); err != nil {
		return nil, err
	} else if err = json.JsonHandler.Unmarshal(bytes, &v); err != nil {
		return nil, err
	} else {
		return v, nil
	}
}
