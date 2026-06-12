#!/bin/bash
# Audit Script for Stub Detection (Linux/macOS)
# Purpose: Identify all stub/placeholder/partial implementations in faster-blaster-reference
# Usage: chmod +x audit_stubs_linux.sh && ./audit_stubs_linux.sh

SOURCE_DIR="${1:-.}/faster-blaster-reference/src"
OUTPUT_FILE="IMPLEMENTATION_STATUS_AUDIT.csv"

echo "🔍 Scanning for stubs in: $SOURCE_DIR"

# Initialize CSV header
{
    echo "operation,file,status,completeness_percent,category,priority"
    
    # Find all .c files
    find "$SOURCE_DIR" -name "*.c" -type f | while read -r file; do
        
        # Extract operation name from filename
        operation=$(basename "$file" .c)
        
        # Determine category from path
        if [[ "$file" =~ blas.*l1 ]]; then
            category="BLAS L1"
            priority=1
        elif [[ "$file" =~ blas.*l2 ]]; then
            category="BLAS L2"
            priority=2
        elif [[ "$file" =~ blas.*l3 ]]; then
            category="BLAS L3"
            priority=3
        elif [[ "$file" =~ lapack.*driver ]]; then
            category="LAPACK Driver"
            priority=4
        elif [[ "$file" =~ lapack.*computational ]]; then
            category="LAPACK Computational"
            priority=5
        elif [[ "$file" =~ lapack.*auxiliary ]]; then
            category="LAPACK Auxiliary"
            priority=6
        else
            category="Other"
            priority=7
        fi
        
        # Check file size (bytes)
        file_size=$(wc -c < "$file")
        line_count=$(wc -l < "$file")
        
        # Determine status
        status="COMPLETE"
        completeness=100
        
        # Check for stub patterns
        if grep -q "TODO\|FIXME\|NOT_IMPLEMENTED\|STUB\|placeholder" "$file" 2>/dev/null; then
            status="PARTIAL"
            completeness=30
        fi
        
        # Check for empty function bodies (very crude check)
        if grep -E "^[[:space:]]*\{[[:space:]]*\}[[:space:]]*$" "$file" >/dev/null 2>&1; then
            status="EMPTY"
            completeness=0
        fi
        
        # Check if very small file (likely stub)
        if [[ $line_count -lt 15 ]] && [[ "$status" == "COMPLETE" ]]; then
            status="REVIEW"
            completeness=50
        fi
        
        # Check for return statements without implementation
        if grep -q "return;\|return 0;\|return NULL;" "$file" 2>/dev/null && \
           ! grep -q "for\|while\|if\|switch" "$file" 2>/dev/null; then
            status="EMPTY"
            completeness=0
        fi
        
        # Output CSV line
        echo "$operation,$file,$status,$completeness,$category,$priority"
        
        # Progress indicator
        case $status in
            "COMPLETE")
                printf "  %-40s : %s (%d%%)\n" "$operation" "$status" "$completeness"
                ;;
            "PARTIAL"|"REVIEW")
                printf "  %-40s : %s (%d%%)\n" "$operation" "$status" "$completeness"
                ;;
            "EMPTY")
                printf "  %-40s : %s (%d%%)\n" "$operation" "$status" "$completeness"
                ;;
        esac
    done
} > "$OUTPUT_FILE"

echo ""
echo "✅ Audit complete! Results saved to: $OUTPUT_FILE"
echo ""
echo "📊 Summary:"
echo "  Complete: $(grep -c ",COMPLETE," "$OUTPUT_FILE")"
echo "  Partial:  $(grep -c ",PARTIAL," "$OUTPUT_FILE")"
echo "  Empty:    $(grep -c ",EMPTY," "$OUTPUT_FILE")"
echo "  Review:   $(grep -c ",REVIEW," "$OUTPUT_FILE")"
echo ""
echo "Next Steps:"
echo "  1. Review $OUTPUT_FILE"
echo "  2. Sort by completeness_percent to prioritize work"
echo "  3. Assign operations to developers"
echo "  4. Begin implementation with highest priority items"
