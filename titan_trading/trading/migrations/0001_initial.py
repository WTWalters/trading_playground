"""
Initial migration for trading app.

This migration creates the necessary database schema.
"""
from django.db import migrations


class Migration(migrations.Migration):
    initial = True
    
    dependencies = [
    ]

    operations = [
        # Check if TimescaleDB is installed and enabled
        migrations.RunSQL(
            sql="""
            DO $$
            BEGIN
                -- Skip operations if TimescaleDB extension is not installed
                IF EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
                ) THEN
                    -- Only run TimescaleDB operations if tables exist
                    IF EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'trading_price'
                    ) THEN
                        -- Create hypertable for Price model
                        PERFORM create_hypertable('trading_price', 'timestamp', 
                                               chunk_time_interval => interval '1 day',
                                               if_not_exists => TRUE);
                        
                        -- Set up compression policy
                        ALTER TABLE trading_price SET (
                            timescaledb.compress,
                            timescaledb.compress_segmentby = 'symbol_id'
                        );
                        
                        PERFORM add_compression_policy('trading_price', interval '7 days');
                        
                        -- Create continuous aggregates
                        CREATE MATERIALIZED VIEW IF NOT EXISTS trading_daily_prices
                        WITH (timescaledb.continuous) AS
                        SELECT
                            time_bucket('1 day', timestamp) AS day,
                            symbol_id,
                            first(open, timestamp) AS open,
                            max(high) AS high,
                            min(low) AS low,
                            last(close, timestamp) AS close,
                            sum(volume) AS volume,
                            count(*) AS sample_count
                        FROM trading_price
                        GROUP BY day, symbol_id;
                        
                        PERFORM add_continuous_aggregate_policy('trading_daily_prices',
                            start_offset => NULL,
                            end_offset => interval '1 hour',
                            schedule_interval => interval '1 day');
                    END IF;
                END IF;
            END
            $$;
            """,
            reverse_sql="SELECT 1;"  # No easy way to reverse this
        ),
    ]
