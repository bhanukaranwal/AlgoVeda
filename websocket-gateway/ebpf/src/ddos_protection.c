/*!
 * eBPF-based DDoS Protection for AlgoVeda WebSocket Gateway
 * Kernel-space packet filtering with sub-microsecond decisions
 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

#define MAX_MAP_ENTRIES 1000000
#define RATE_LIMIT_THRESHOLD 1000  // packets per second
#define TIME_WINDOW 1000000000     // 1 second in nanoseconds

// Map to track connection rates per IP
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, MAX_MAP_ENTRIES);
    __type(key, __u32);    // IP address
    __type(value, struct rate_limit_info);
} rate_limit_map SEC(".maps");

// Map for storing DDoS detection rules
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1000);
    __type(key, __u32);    // Rule ID
    __type(value, struct ddos_rule);
} ddos_rules_map SEC(".maps");

// Map for blocked IPs
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 100000);
    __type(key, __u32);    // IP address
    __type(value, __u64);  // Block expiry time
} blocked_ips_map SEC(".maps");

struct rate_limit_info {
    __u64 last_packet_time;
    __u32 packet_count;
    __u32 byte_count;
    __u16 syn_count;       // Track SYN flood attempts
    __u16 connection_count;
};

struct ddos_rule {
    __u32 max_pps;         // Max packets per second
    __u32 max_bps;         // Max bytes per second
    __u16 max_connections; // Max concurrent connections
    __u16 max_syn_rate;    // Max SYN packets per second
    __u32 block_duration;  // Block duration in seconds
};

static __always_inline int parse_tcp(void *data, void *data_end, 
                                    struct tcphdr **tcph) {
    struct tcphdr *tcp = data;
    
    if ((void *)(tcp + 1) > data_end)
        return -1;
    
    *tcph = tcp;
    return 0;
}

static __always_inline int parse_ip(void *data, void *data_end,
                                   struct iphdr **iph) {
    struct iphdr *ip = data;
    
    if ((void *)(ip + 1) > data_end)
        return -1;
    
    if (ip->version != 4)
        return -1;
    
    *iph = ip;
    return 0;
}

SEC("xdp")
int ddos_protection(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;
    
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;
    
    // Only process IPv4 packets
    if (bpf_ntohs(eth->h_proto) != ETH_P_IP)
        return XDP_PASS;
    
    struct iphdr *ip;
    if (parse_ip(data + sizeof(*eth), data_end, &ip) < 0)
        return XDP_PASS;
    
    __u32 src_ip = bpf_ntohl(ip->saddr);
    __u64 now = bpf_ktime_get_ns();
    
    // Check if IP is already blocked
    __u64 *block_expiry = bpf_map_lookup_elem(&blocked_ips_map, &src_ip);
    if (block_expiry && *block_expiry > now) {
        return XDP_DROP;  // Still blocked
    }
    
    // Get or create rate limit info for this IP
    struct rate_limit_info *rate_info = 
        bpf_map_lookup_elem(&rate_limit_map, &src_ip);
    
    struct rate_limit_info new_rate_info = {};
    if (!rate_info) {
        new_rate_info.last_packet_time = now;
        new_rate_info.packet_count = 1;
        new_rate_info.byte_count = bpf_ntohs(ip->tot_len);
        rate_info = &new_rate_info;
    } else {
        // Reset counters if time window expired
        if (now - rate_info->last_packet_time > TIME_WINDOW) {
            rate_info->packet_count = 1;
            rate_info->byte_count = bpf_ntohs(ip->tot_len);
            rate_info->syn_count = 0;
            rate_info->last_packet_time = now;
        } else {
            rate_info->packet_count++;
            rate_info->byte_count += bpf_ntohs(ip->tot_len);
        }
    }
    
    // Check for TCP SYN flood
    if (ip->protocol == IPPROTO_TCP) {
        struct tcphdr *tcp;
        if (parse_tcp(data + sizeof(*eth) + (ip->ihl * 4), data_end, &tcp) == 0) {
            if (tcp->syn && !tcp->ack) {
                rate_info->syn_count++;
            }
        }
    }
    
    // Get DDoS protection rules (using rule ID 1 for simplicity)
    __u32 rule_id = 1;
    struct ddos_rule *rule = bpf_map_lookup_elem(&ddos_rules_map, &rule_id);
    if (!rule) {
        // Default rule if not found
        struct ddos_rule default_rule = {
            .max_pps = RATE_LIMIT_THRESHOLD,
            .max_bps = 10000000,  // 10MB/s
            .max_connections = 100,
            .max_syn_rate = 50,
            .block_duration = 300  // 5 minutes
        };
        rule = &default_rule;
    }
    
    // Check rate limits
    bool should_block = false;
    
    if (rate_info->packet_count > rule->max_pps) {
        should_block = true;
    }
    
    if (rate_info->byte_count > rule->max_bps) {
        should_block = true;
    }
    
    if (rate_info->syn_count > rule->max_syn_rate) {
        should_block = true;
    }
    
    if (should_block) {
        // Block this IP for the specified duration
        __u64 block_until = now + ((__u64)rule->block_duration * 1000000000);
        bpf_map_update_elem(&blocked_ips_map, &src_ip, &block_until, BPF_ANY);
        
        return XDP_DROP;
    }
    
    // Update rate limit info in map
    bpf_map_update_elem(&rate_limit_map, &src_ip, rate_info, BPF_ANY);
    
    return XDP_PASS;
}

// Program for updating DDoS rules from userspace
SEC("xdp")
int update_ddos_rules(struct xdp_md *ctx) {
    // This would be called from userspace to update rules
    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";
