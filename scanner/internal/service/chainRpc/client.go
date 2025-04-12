package chainRpc

import (
	"context"
	"math/big"

	"github.com/ethereum/go-ethereum/core/types"
	client "github.com/ethereum/go-ethereum/ethclient"
	"github.com/ethereum/go-ethereum/rpc"
)

type Client struct {
	client    *client.Client
	signer    types.Signer
	networkID *big.Int
	url       string
}

func NewClient(url string) *Client {
	c := &Client{
		url: url,
	}

	rpc_, err := rpc.Dial(url)
	if err != nil {
		panic(err)
	}

	c.client = client.NewClient(rpc_)

	if c.networkID, err = c.client.NetworkID(context.Background()); err != nil {
		panic(err)
	}

	c.signer = types.NewEIP155Signer(c.networkID)

	return c
}

func (c *Client) GetNetworkID() *big.Int {
	return c.networkID
}
