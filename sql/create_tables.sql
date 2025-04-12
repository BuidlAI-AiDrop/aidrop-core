-- 블록체인 트랜잭션 테이블
CREATE TABLE IF NOT EXISTS blockchain_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chain_id VARCHAR(50) NOT NULL,
    hash VARCHAR(255) NOT NULL,
    block_number BIGINT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sender VARCHAR(255) NOT NULL,
    receiver VARCHAR(255),
    value NUMERIC(78, 0) DEFAULT 0,
    gas_used NUMERIC(78, 0) DEFAULT 0,
    gas_price NUMERIC(78, 0) DEFAULT 0,
    is_error BOOLEAN DEFAULT FALSE,
    method_id VARCHAR(10),
    method_name VARCHAR(255),
    input_data TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(chain_id, hash)
);

-- 토큰 전송 테이블
CREATE TABLE IF NOT EXISTS token_transfers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chain_id VARCHAR(50) NOT NULL,
    transaction_hash VARCHAR(255) NOT NULL,
    block_number BIGINT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    contract_address VARCHAR(255) NOT NULL,
    token_name VARCHAR(255),
    token_symbol VARCHAR(255),
    token_decimals INTEGER DEFAULT 18,
    from_address VARCHAR(255) NOT NULL,
    to_address VARCHAR(255) NOT NULL,
    value NUMERIC(78, 0) DEFAULT 0,
    token_id VARCHAR(255), -- ERC-721 토큰 ID (NFT)
    token_type VARCHAR(50) DEFAULT 'ERC20', -- ERC20, ERC721, ERC1155 등
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 컨트랙트 상호작용 테이블
CREATE TABLE IF NOT EXISTS contract_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chain_id VARCHAR(50) NOT NULL,
    transaction_hash VARCHAR(255) NOT NULL,
    block_number BIGINT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_address VARCHAR(255) NOT NULL,
    contract_address VARCHAR(255) NOT NULL,
    contract_name VARCHAR(255),
    contract_type VARCHAR(50), -- defi, nft, gaming, social, governance, other 등
    method_id VARCHAR(10),
    method_name VARCHAR(255),
    params TEXT, -- JSON 형식으로 저장된 파라미터
    event_count INTEGER DEFAULT 0,
    is_success BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 토큰 잔액 테이블
CREATE TABLE IF NOT EXISTS token_balances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chain_id VARCHAR(50) NOT NULL,
    address VARCHAR(255) NOT NULL,
    contract_address VARCHAR(255) NOT NULL,
    token_name VARCHAR(255),
    token_symbol VARCHAR(255),
    token_decimals INTEGER DEFAULT 18,
    balance NUMERIC(78, 0) DEFAULT 0,
    token_type VARCHAR(50) DEFAULT 'ERC20', -- ERC20, ERC721, ERC1155 등
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(chain_id, address, contract_address)
);

-- NFT 보유 테이블
CREATE TABLE IF NOT EXISTS nft_holdings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chain_id VARCHAR(50) NOT NULL,
    contract_address VARCHAR(255) NOT NULL,
    token_id VARCHAR(255) NOT NULL,
    owner_address VARCHAR(255) NOT NULL,
    collection_name VARCHAR(255),
    token_uri TEXT,
    metadata TEXT, -- JSON 형식으로 저장된 메타데이터
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(chain_id, contract_address, token_id)
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_tx_sender ON blockchain_transactions (sender, chain_id);
CREATE INDEX IF NOT EXISTS idx_tx_receiver ON blockchain_transactions (receiver, chain_id);
CREATE INDEX IF NOT EXISTS idx_tx_timestamp ON blockchain_transactions (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_tt_from ON token_transfers (from_address, chain_id);
CREATE INDEX IF NOT EXISTS idx_tt_to ON token_transfers (to_address, chain_id);
CREATE INDEX IF NOT EXISTS idx_tt_timestamp ON token_transfers (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_ci_user ON contract_interactions (user_address, chain_id);
CREATE INDEX IF NOT EXISTS idx_ci_contract ON contract_interactions (contract_address, chain_id);
CREATE INDEX IF NOT EXISTS idx_ci_type ON contract_interactions (contract_type);

CREATE INDEX IF NOT EXISTS idx_tb_address ON token_balances (address, chain_id);
CREATE INDEX IF NOT EXISTS idx_nh_owner ON nft_holdings (owner_address, chain_id); 