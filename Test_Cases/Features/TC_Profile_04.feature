Feature: ZARA Profile Page Simulation

  Scenario: Try accessing profile page
    Given the user is on the ZARA homepage
    When the user enters an valid email and password
    When the user clicks on the profile icon
    Then the user should be able to logout from ZARA
