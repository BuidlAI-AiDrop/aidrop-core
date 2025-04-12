-- API 요청 테이블
CREATE TABLE IF NOT EXISTS analysis_requests (
  request_id text PRIMARY KEY,
  source_address text NOT NULL,
  story_address text NOT NULL,
  source_chain_id text NOT NULL,
  status text NOT NULL DEFAULT 'pending',
  message text,
  image_url text,
  user_type text,
  error text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now()
);

-- 분석 결과 테이블
CREATE TABLE IF NOT EXISTS analysis_results (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  request_id text REFERENCES analysis_requests(request_id),
  source_address text NOT NULL,
  story_address text NOT NULL,
  source_chain_id text NOT NULL,
  image_url text,
  user_type text,
  cluster integer,
  traits jsonb,
  full_result jsonb,
  status text NOT NULL DEFAULT 'completed',
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now()
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_requests_addresses ON analysis_requests(source_address, story_address, source_chain_id);
CREATE INDEX IF NOT EXISTS idx_results_addresses ON analysis_results(source_address, story_address, source_chain_id);
CREATE INDEX IF NOT EXISTS idx_requests_status ON analysis_requests(status);
CREATE INDEX IF NOT EXISTS idx_results_status ON analysis_results(status);

-- 권한 설정
ALTER TABLE analysis_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;

-- 모든 사용자 인증 정책
CREATE POLICY "모든 인증된 사용자가 요청을 조회할 수 있음" ON analysis_requests
  FOR SELECT USING (auth.role() = 'authenticated');

CREATE POLICY "모든 인증된 사용자가 결과를 조회할 수 있음" ON analysis_results
  FOR SELECT USING (auth.role() = 'authenticated');

CREATE POLICY "서비스 계정만 요청을 생성할 수 있음" ON analysis_requests
  FOR INSERT WITH CHECK (auth.role() = 'service_role');
  
CREATE POLICY "서비스 계정만 요청을 수정할 수 있음" ON analysis_requests
  FOR UPDATE USING (auth.role() = 'service_role');

CREATE POLICY "서비스 계정만 결과를 생성할 수 있음" ON analysis_results
  FOR INSERT WITH CHECK (auth.role() = 'service_role');
  
CREATE POLICY "서비스 계정만 결과를 수정할 수 있음" ON analysis_results
  FOR UPDATE USING (auth.role() = 'service_role');

-- 뷰 생성 (요청과 결과 조인)
CREATE OR REPLACE VIEW analysis_summary AS
  SELECT 
    r.request_id,
    r.source_address,
    r.story_address,
    r.source_chain_id,
    r.status as request_status,
    r.created_at as requested_at,
    r.updated_at as request_updated_at,
    res.id as result_id,
    res.image_url,
    res.user_type,
    res.cluster,
    res.traits,
    res.status as result_status,
    res.created_at as completed_at
  FROM analysis_requests r
  LEFT JOIN analysis_results res ON r.request_id = res.request_id
  ORDER BY r.created_at DESC; 