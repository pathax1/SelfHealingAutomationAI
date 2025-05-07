Feature: IKEA Search Functionality

  Scenario: Search for a product
    Given the user is on the IKEA homepage
    When the user searches for "chair"
    Then the results should include items related to "chair"
