class NetworkAnalyzer:

    @staticmethod
    def analyze(results):
        if not results:
            return {}

        latencies = [
            result["latency_us"]
            for result in results
        ]

        queueing_delays = [
            result["start_time_us"]
            - result["arrival_time_us"]
            for result in results
        ]

        deadline_misses = sum(
            not result["deadline_met"]
            for result in results
        )

        jitter = 0

        if len(latencies) > 1:
            differences = [
                abs(latencies[i] - latencies[i - 1])
                for i in range(1, len(latencies))
            ]

            jitter = sum(differences) / len(differences)

        return {
            "average_latency_us":
                sum(latencies) / len(latencies),

            "maximum_latency_us":
                max(latencies),

            "minimum_latency_us":
                min(latencies),

            "average_queueing_delay_us":
                sum(queueing_delays)
                / len(queueing_delays),

            "jitter_us":
                jitter,

            "deadline_misses":
                deadline_misses,

            "total_flows":
                len(results)
        }