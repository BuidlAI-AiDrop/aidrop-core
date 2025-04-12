-- 테스트 트랜잭션 데이터 삽입
INSERT INTO blockchain_transactions 
(chain_id, hash, block_number, timestamp, sender, receiver, value, gas_used, gas_price, is_error, method_id, method_name)
VALUES
('1', '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef', 12345678, '2025-01-15 12:30:45+00', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0xa1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4', 100000000000000000, 21000, 20000000000, FALSE, '0xa9059cbb', 'transfer'),
('1', '0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890', 12345680, '2025-01-16 14:22:33+00', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0xb2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f6', 250000000000000000, 31000, 25000000000, FALSE, '0x', NULL),
('1', '0x7890abcdef1234567890abcdef1234567890abcdef1234567890abcdef123456', 12345685, '2025-01-17 09:15:22+00', '0x5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890a1', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', 500000000000000000, 21000, 20000000000, FALSE, '0x', NULL),
('1', '0x4567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef123', 12345690, '2025-01-18 16:45:12+00', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0x0bEF27FEB58e857046d630B2c03dFb7bae567494', 10000000000000000, 150000, 30000000000, FALSE, '0x095ea7b3', 'approve');

-- 테스트 토큰 전송 데이터 삽입
INSERT INTO token_transfers
(chain_id, transaction_hash, block_number, timestamp, contract_address, token_name, token_symbol, token_decimals, from_address, to_address, value, token_type)
VALUES
('1', '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef', 12345678, '2025-01-15 12:30:45+00', '0x6b175474e89094c44da98b954eedeac495271d0f', 'Dai Stablecoin', 'DAI', 18, '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0xa1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4', 100000000000000000000, 'ERC20'),
('1', '0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890', 12345680, '2025-01-16 14:22:33+00', '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48', 'USD Coin', 'USDC', 6, '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0xb2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f6', 50000000, 'ERC20'),
('1', '0x90abcdef1234567890abcdef1234567890abcdef1234567890abcdef12345678', 12345683, '2025-01-16 18:10:05+00', '0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d', 'Bored Ape Yacht Club', 'BAYC', 0, '0xd4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', 1, 'ERC721');

-- 테스트 컨트랙트 상호작용 데이터 삽입
INSERT INTO contract_interactions
(chain_id, transaction_hash, block_number, timestamp, user_address, contract_address, contract_name, contract_type, method_id, method_name, params, event_count, is_success)
VALUES
('1', '0x4567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef123', 12345690, '2025-01-18 16:45:12+00', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0x0bEF27FEB58e857046d630B2c03dFb7bae567494', 'ENS DAO', 'social', '0xdf8de3e7', 'vote', '{"proposal": 41, "support": 1}', 2, TRUE),
('1', '0x67890abcdef1234567890abcdef1234567890abcdef1234567890abcdef12345', 12345692, '2025-01-19 10:22:18+00', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D', 'Uniswap V2 Router', 'defi', '0x38ed1739', 'swapExactTokensForTokens', '{"amountIn": "1000000000000000000", "amountOutMin": "950000"}', 3, TRUE),
('1', '0x90abcdef1234567890abcdef1234567890abcdef1234567890abcdef12345678', 12345695, '2025-01-20 14:35:42+00', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d', 'Bored Ape Yacht Club', 'nft', '0x6352211e', 'ownerOf', '{"tokenId": "1234"}', 1, TRUE),
('1', '0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890', 12345698, '2025-01-21 08:12:55+00', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0x86935F11C86623deC8a25696E1C19a8659CbF95d', 'Aavegotchi', 'gaming', '0xa0712d68', 'mint', '{"tokenId": 12345, "uri": "ipfs://..."}', 2, TRUE);

-- 테스트 토큰 잔액 데이터 삽입
INSERT INTO token_balances
(chain_id, address, contract_address, token_name, token_symbol, token_decimals, balance, token_type, last_updated)
VALUES
('1', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0x6b175474e89094c44da98b954eedeac495271d0f', 'Dai Stablecoin', 'DAI', 18, 250000000000000000000, 'ERC20', '2025-01-21 08:12:55+00'),
('1', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48', 'USD Coin', 'USDC', 6, 75000000, 'ERC20', '2025-01-21 08:12:55+00'),
('1', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', 'Wrapped Ether', 'WETH', 18, 5000000000000000000, 'ERC20', '2025-01-21 08:12:55+00');

-- 테스트 NFT 보유 데이터 삽입
INSERT INTO nft_holdings
(chain_id, contract_address, token_id, owner_address, collection_name, token_uri, metadata)
VALUES
('1', '0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d', '1234', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', 'Bored Ape Yacht Club', 'ipfs://QmeSjSinHpPnmXmspMjwiXyN6zS4E9zccariGR3jxcaWtq/1234', '{"image":"ipfs://QmXBQhQvi6BqrDNokFvvo8Rfm42KFhhDADNpNaJPGNonrW","attributes":[{"trait_type":"Background","value":"Orange"},{"trait_type":"Eyes","value":"Bored"}]}'),
('1', '0x86935F11C86623deC8a25696E1C19a8659CbF95d', '12345', '0xf620c88414596cb9ab354bb347a814c1f0f078ca', 'Aavegotchi', 'ipfs://QmPQdVU1riwzijhCs5Kw7mFxQJ87uLmRyRVpFCwGwLeLnM/12345', '{"name":"Aavegotchi #12345","description":"Spooky ghost friend","image":"ipfs://QmUVEBPab1kcYBNdFWYMeTKghGjA18qVe7Z8CQUbYs3VwH"}'); 