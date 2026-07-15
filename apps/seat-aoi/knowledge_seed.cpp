#include "knowledge_seed.h"

#include <sai/knowledge/knowledge_graph.h>
#include <sai/knowledge/knowledge_record.h>

// Populate KnowledgeGraph with sample seat leather inspection domain data.
// In production, this comes from MES/PLC via OPC UA.
auto SeedKnowledgeGraph(sai::knowledge::KnowledgeGraph& kg) -> void {
    using namespace sai::knowledge;

    auto make_record = [](std::initializer_list<std::pair<const char*, FieldValue>> items) {
        KnowledgeRecord r;
        for (auto& [k, v] : items) r.fields[k] = v;
        return r;
    };

    // --- Material nodes ---
    auto mat = kg.InsertNode("Material",
        make_record({{"name", std::string("Nappa_Leather_Black")},
                     {"sku", std::string("SKU-NLB-001")},
                     {"thickness_mm", 1.2}}));
    auto mat2 = kg.InsertNode("Material",
        make_record({{"name", std::string("Nappa_Leather_Brown")},
                     {"sku", std::string("SKU-NLB-002")},
                     {"thickness_mm", 1.2}}));

    // --- Supplier nodes ---
    auto supp = kg.InsertNode("Supplier",
        make_record({{"name", std::string("LeatherWorks_GmbH")},
                     {"location", std::string("Stuttgart_DE")},
                     {"certification", std::string("ISO_9001")}}));
    auto supp2 = kg.InsertNode("Supplier",
        make_record({{"name", std::string("PremiumHide_Ltd")},
                     {"location", std::string("Modena_IT")},
                     {"certification", std::string("ISO_9001")}}));

    // --- Batch nodes ---
    auto batch1 = kg.InsertNode("Batch",
        make_record({{"batch_id", std::string("B2026-001")},
                     {"reject_rate_pct", 0.8},
                     {"total_units", static_cast<std::int64_t>(5000)},
                     {"surface", std::string("seat_leather_driver")}}));
    auto batch2 = kg.InsertNode("Batch",
        make_record({{"batch_id", std::string("B2026-002")},
                     {"reject_rate_pct", 4.2},
                     {"total_units", static_cast<std::int64_t>(3200)},
                     {"surface", std::string("seat_leather_passenger")}}));
    auto batch3 = kg.InsertNode("Batch",
        make_record({{"batch_id", std::string("B2026-003")},
                     {"reject_rate_pct", 1.5},
                     {"total_units", static_cast<std::int64_t>(8000)},
                     {"surface", std::string("seat_leather_driver")}}));

    // --- Edges ---
    (void)kg.InsertEdge(supp.value(), mat.value(), "SUPPLIES",
        make_record({{"lead_time_days", static_cast<std::int64_t>(14)}}));
    (void)kg.InsertEdge(supp2.value(), mat2.value(), "SUPPLIES",
        make_record({{"lead_time_days", static_cast<std::int64_t>(21)}}));
    (void)kg.InsertEdge(mat.value(), batch1.value(), "PRODUCED_AS",
        make_record({{"production_line", std::string("L3")}}));
    (void)kg.InsertEdge(mat.value(), batch3.value(), "PRODUCED_AS",
        make_record({{"production_line", std::string("L3")}}));
    (void)kg.InsertEdge(mat2.value(), batch2.value(), "PRODUCED_AS",
        make_record({{"production_line", std::string("L5")}}));
}
